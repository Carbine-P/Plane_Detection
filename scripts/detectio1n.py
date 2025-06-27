#!/usr/bin/env python3
import socket
import struct
import argparse
import time
import threading
import numpy as np # 需要 numpy 来处理向量

# --- ROS 相关导入 (条件性) ---
try:
    import rospy
    from geometry_msgs.msg import Quaternion
    from plane_detection.msg import PlaneCorners  # 添加角点消息类型导入
    # 尝试导入 tf 用于转换，如果失败则使用 numpy
    try:
        import tf.transformations as tf_trans
        USE_TF_TRANS = True
    except ImportError:
        print("[警告] 未找到 tf 库 (python-tf)。将使用纯 Numpy 进行四元数运算。")
        USE_TF_TRANS = False
    ROS_AVAILABLE = True
except ImportError:
    print("[信息] ROS 相关库未找到或未安装。将无法使用 --ros 选项。")
    ROS_AVAILABLE = False
    # 定义占位符，以防代码引用它们
    class PlaneCorners:
        pass

# --- 配置 ---
DEFAULT_LOCAL_IP = '172.17.30.74'  # 修改为本地设备IP，不再监听所有网络接口
DEFAULT_LOCAL_PORT = 6767     # 使用UDP端口6767
DEFAULT_TARGET_PORT = 6767    # 目标设备端口6767
DEFAULT_TARGET_IP = '172.17.30.64'  # 目标设备IP (walking computer) - 修改为正确的目标IP
LOCAL_INTERFACE_IP = '172.17.30.74' # 本地设备IP (detection computer)
DEFAULT_SEND_INTERVAL = 1.0   # 主动发送模式的间隔(秒)

# --- 协议常量 ---
PROTOCOL_TYPE_REQUEST = 0xC1  # 请求包
PROTOCOL_TYPE_RESPONSE = 0xC2 # 响应包
SOURCE_CODE_DETECTION = 0x33  # 探测子系统代码
TARGET_CODE_WALKING = 0x55    # 行走子系统代码
DATA_TYPE = 0xCF              # 数据类型
STATUS_BROADCAST = 0x00       # 状态广播

# --- 共享数据与锁 ---
data_lock = threading.Lock()
# 用于存储从 ROS 获取的最新法向量 (世界坐标系 Z 轴)
latest_normal_vector = np.array([0.0, 0.0, 1.0]) # 默认 Z 轴向上 (ROS标准)

# --- 动态角点数据 ---
# 用于存储从平面检测节点获取的动态角点数据
latest_corners = {
    'corner1': (50.0, -2.0, 0.0),  # 默认值，将被平面检测结果覆盖
    'corner2': (50.0, 2.0, 0.0),
    'corner3': (30.0, 2.0, 0.0),
    'corner4': (30.0, -2.0, 0.0)
}
has_dynamic_corners = False  # 标志是否收到动态角点数据

# --- 静态角点数据 ---
# 默认角点坐标 (单位: cm)
STATIC_NEAR_LEFT = (50.0, -2.0, 0.0)
STATIC_NEAR_RIGHT = (50.0, 2.0, 0.0)
STATIC_FAR_RIGHT = (30.0, 2.0, 0.0)
STATIC_FAR_LEFT = (30.0, -2.0, 0.0)

# --- 便利函数 ---
def hex_format(data, bytes_per_group=2, max_bytes_per_line=16):
    """格式化二进制数据为易读的十六进制格式"""
    if not data:
        return "[空数据]"
        
    hex_data = data.hex()
    result = []
    
    # 每 bytes_per_group 个字节分组
    hex_groups = [hex_data[i:i+bytes_per_group*2] for i in range(0, len(hex_data), bytes_per_group*2)]
    
    # 每行显示 max_bytes_per_line 个字节
    for i in range(0, len(hex_groups), max_bytes_per_line//bytes_per_group):
        line_groups = hex_groups[i:i + max_bytes_per_line//bytes_per_group]
        offset = i * bytes_per_group
        # 添加行偏移和十六进制数据
        line = f"    [{offset:04X}] " + " ".join(line_groups)
        result.append(line)
        
    return "\n".join(result)

def print_network_interfaces():
    """打印所有网络接口信息用于调试"""
    try:
        import psutil
        print("\n[网络接口信息]")
        for iface, addr_list in psutil.net_if_addrs().items():
            for addr in addr_list:
                if addr.family == socket.AF_INET:
                    print(f"  接口: {iface}, IP: {addr.address}")
        print("")
    except ImportError:
        print("[警告] 无法导入psutil库，跳过网络接口信息显示")
    except Exception as e:
        print(f"[警告] 获取网络接口信息失败: {e}")

def create_udp_socket(bind_ip=DEFAULT_LOCAL_IP, bind_port=DEFAULT_LOCAL_PORT, reuse_addr=True, receiver=True):
    """创建并配置UDP socket"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 设置超时和重用地址
    if receiver:
        sock.settimeout(0.1)  # 接收socket设置超时以避免阻塞
    if reuse_addr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind((bind_ip, bind_port))
        print(f"[网络] Socket 绑定到 {bind_ip}:{bind_port}" + (" [接收]" if receiver else " [发送]"))
        return sock
    except Exception as e:
        print(f"[错误] 绑定Socket到 {bind_ip}:{bind_port} 失败: {e}")
        sock.close()
        # 如果是发送socket，尝试不绑定端口
        if not receiver:
            print("[网络] 尝试使用系统分配的随机端口...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if reuse_addr:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock
        return None

# --- 解析请求包 (0xC1) ---
def parse_request_packet(data):
    """解析 0xC1 请求数据包 (小端格式)"""
    # 格式: <hhhhhhI (6 short, 1 unsigned int) - 小端格式
    header_format = '<hhhhhhI'
    expected_size = struct.calcsize(header_format)

    if len(data) < expected_size:
        print(f"  [警告] 接收请求包大小不足 ({len(data)} 字节), 期望 {expected_size} 字节")
        return None

    try:
        protocol_type, source_code, local_code, target_code, data_type, data_flag, packet_count = \
            struct.unpack(header_format, data[:expected_size])

        # 检查协议类型
        if protocol_type != PROTOCOL_TYPE_REQUEST:
            print(f"  [警告] 收到错误的协议类型 (期望 0x{PROTOCOL_TYPE_REQUEST:04X}, 收到 0x{protocol_type:04X})")
            # 依然返回解析结果，只是打印警告
            
        return {
            'protocol_type': protocol_type,
            'source_code': source_code,
            'local_code': local_code,
            'target_code': target_code,
            'data_type': data_type,
            'data_flag': data_flag,
            'packet_count': packet_count
        }

    except struct.error as e:
        print(f"  [错误] 解析请求包失败 (struct error): {e}")
        print(f"    数据包大小: {len(data)}, 内容: {data.hex()}")
        return None
    except Exception as e:
        print(f"  [错误] 解析请求包时发生未知错误: {e}")
        return None

# --- 平面检测回调函数 ---
def plane_detection_callback(msg):
    """处理平面检测结果消息，更新角点数据"""
    global latest_corners, has_dynamic_corners
    
    try:
        # 提取角点数据 (已经是cm单位，直接使用)
        corners = []
        for point in msg.corners:
            # 确保使用float32格式
            x_cm = float(point.x)  # 已经是厘米单位
            y_cm = float(point.y)
            z_cm = float(point.z)
            corners.append((x_cm, y_cm, z_cm))
        
        # 更新全局角点数据
        with data_lock:
            if len(corners) >= 4:
                latest_corners['corner1'] = corners[0]
                latest_corners['corner2'] = corners[1] 
                latest_corners['corner3'] = corners[2]
                latest_corners['corner4'] = corners[3]
                has_dynamic_corners = True
                
                # 直接使用plane_detector_node.py计算的精确法向量 (基于RANSAC)
                normal = msg.normal
                latest_normal_vector[:] = [normal.x, normal.y, normal.z]
                print(f"[平面检测回调] 使用RANSAC拟合的法向量: [{normal.x:.3f}, {normal.y:.3f}, {normal.z:.3f}]")
                
                # 数据已更新，无需额外输出（避免重复显示）
            else:
                print(f"[平面检测回调] [警告] 收到的角点数量不足: {len(corners)}个")
                
    except Exception as e:
        print(f"[平面检测回调] [错误] 处理平面检测消息时出错: {e}")

# --- ROS 初始化 ---
def setup_ros():
    """初始化 ROS 节点和平面检测订阅者"""
    if not ROS_AVAILABLE:
        print("[错误] ROS 功能不可用。请确保已安装 ROS 相关库 (rospy, plane_detection.msg)。")
        return False
    try:
        print("[ROS] 初始化节点 'detection_computer_plane_listener'...")
        rospy.init_node('detection_computer_plane_listener', anonymous=True)
        
        # 订阅平面检测结果话题
        plane_detection_topic = '/result'
        print(f"[ROS] 订阅平面检测话题 '{plane_detection_topic}'...")
        rospy.Subscriber(plane_detection_topic, PlaneCorners, plane_detection_callback)
        
        print(f"[ROS] 成功订阅 '{plane_detection_topic}'")
        return True
    except Exception as e:
        print(f"[ROS] [错误] 设置失败: {e}")
        return False

# --- 创建响应包 (0xC2) ---
def create_response_packet():
    """创建 0xC2 响应包, 使用最新的法向量(若有)和角点数据 (优先使用动态角点) (小端格式)"""
    # 小端格式: <hhhhhhfffffffffffffff
    response_format = '<hhhhhhfffffffffffffff'

    # 获取当前的法向量和角点数据
    with data_lock:
        current_normal = latest_normal_vector
        use_dynamic = has_dynamic_corners
        current_corners = latest_corners.copy()

    # 选择角点数据源
    if use_dynamic:
        # 使用动态检测到的角点
        corner1 = current_corners['corner1']
        corner2 = current_corners['corner2'] 
        corner3 = current_corners['corner3']
        corner4 = current_corners['corner4']
        corner_source = "动态检测"
    else:
        # 使用静态角点作为备用
        corner1 = STATIC_NEAR_LEFT
        corner2 = STATIC_NEAR_RIGHT
        corner3 = STATIC_FAR_RIGHT  
        corner4 = STATIC_FAR_LEFT
        corner_source = "静态默认"

    try:
        packet = struct.pack(response_format,
                            PROTOCOL_TYPE_RESPONSE,
                            SOURCE_CODE_DETECTION,
                            SOURCE_CODE_DETECTION,
                            TARGET_CODE_WALKING,
                            DATA_TYPE,
                            STATUS_BROADCAST,
                            # 当前法向量 (来自 ROS 或默认值)
                            current_normal[0], current_normal[1], current_normal[2],
                            # 角点数据 (动态或静态)
                            corner1[0], corner1[1], corner1[2],
                            corner2[0], corner2[1], corner2[2],
                            corner3[0], corner3[1], corner3[2],
                            corner4[0], corner4[1], corner4[2])
        
        # 打印当前使用的数据源 (调试信息)
        if use_dynamic:
            print(f"[响应包] 使用{corner_source}角点数据")
        
        return packet
    except struct.error as e:
        print(f"[错误] 创建响应包失败: {e}")
        return None

# --- 接收线程函数 ---
def receiver_thread_func(sock, target_ip=None, target_port=DEFAULT_TARGET_PORT):
    """接收请求并回复响应的线程"""
    if target_ip:
        print(f"[接收线程] 启动，仅接收来自 {target_ip} 的数据包")
    else:
        print("[接收线程] 启动，等待来自任意IP的数据包")
    
    # 保存socket绑定的本地IP，用于重新创建socket
    try:
        local_address = sock.getsockname()
        local_ip = local_address[0]
        local_port = local_address[1]
    except:
        # 如果无法获取，使用默认值
        local_ip = DEFAULT_LOCAL_IP
        local_port = DEFAULT_LOCAL_PORT
    
    # 创建用于响应的发送socket - 使用随机端口而不是67端口来发送响应
    try:
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("[网络] 创建发送socket成功 (使用系统分配的随机端口)")
    except Exception as e:
        print(f"[错误] 创建发送socket失败: {e}")
        print("[错误] 无法创建发送socket，将使用接收socket发送响应")
        send_sock = sock
    
    # 接收计数器
    recv_count = 0
    last_status_time = time.time()
    last_debug_time = time.time()
    
    try:
        print(f"[接收线程] 开始监听UDP端口 {DEFAULT_LOCAL_PORT}")
        while True:
            try:
                # 定期显示状态
                current_time = time.time()
                if current_time - last_status_time > 5:
                    print(f"[接收线程] 状态: 已接收 {recv_count} 个数据包")
                    last_status_time = current_time
                
                # 每10秒输出一条调试消息，表明线程仍在运行
                if current_time - last_debug_time > 10:
                    print(f"[接收线程] 正在等待数据包...")
                    last_debug_time = current_time
                
                # 尝试接收数据
                data, addr = sock.recvfrom(2048)
                
                # 如果指定了目标IP，只处理来自该IP的数据包
                if target_ip and addr[0] != target_ip:
                    print(f"\n[接收线程] 忽略来自 {addr[0]} 的数据包，因为它不是指定的目标IP ({target_ip})")
                    continue
                
                recv_count += 1
                
                # 显示接收到的数据
                print(f"\n[接收数据] <- {addr[0]}:{addr[1]} ({len(data)} 字节)")
                print("  原始数据 (十六进制):")
                print(hex_format(data))
                
                # 解析数据包
                request_info = parse_request_packet(data)
                if not request_info:
                    print("  [警告] 无法解析数据包，忽略")
                    continue
                
                # 显示解析结果
                print(f"  协议类型: 0x{request_info['protocol_type']:04X} " +
                      f"({'请求包' if request_info['protocol_type'] == PROTOCOL_TYPE_REQUEST else '未知类型'})")
                print(f"  来源代码: 0x{request_info['source_code']:04X} " +
                      f"({'行走系统' if request_info['source_code'] == TARGET_CODE_WALKING else '未知系统'})")
                print(f"  目标代码: 0x{request_info['target_code']:04X} " +
                      f"({'探测系统' if request_info['target_code'] == SOURCE_CODE_DETECTION else '未知系统'})")
                print(f"  数据类型: 0x{request_info['data_type']:04X}, 标志: 0x{request_info['data_flag']:04X}")
                print(f"  包序号: {request_info['packet_count']}")
                
                # 创建并发送响应
                response_packet = create_response_packet()
                if response_packet:
                    # 发送给指定的目标IP和端口
                    try:
                        # 使用接收到的数据包的源IP作为响应目标 (除非指定了固定目标IP)
                        target_addr = (target_ip if target_ip else addr[0], target_port)
                        print(f"  正在向 {target_addr[0]}:{target_addr[1]} 发送响应...")
                        bytes_sent = send_sock.sendto(response_packet, target_addr)
                        print(f"[发送响应] -> {target_addr[0]}:{target_addr[1]} ({bytes_sent} 字节)")
                        
                        # 打印响应内容
                        print("  响应原始数据 (十六进制):")
                        print(hex_format(response_packet))
                        
                        # 打印法向量和角点信息
                        with data_lock:
                            current_normal = latest_normal_vector
                            use_dynamic = has_dynamic_corners
                            current_corners = latest_corners.copy()
                        
                        print(f"  法向量: [{current_normal[0]:.3f}, {current_normal[1]:.3f}, {current_normal[2]:.3f}]")
                        
                        if use_dynamic:
                            print(f"  角点1: [{current_corners['corner1'][0]:.1f}, {current_corners['corner1'][1]:.1f}, {current_corners['corner1'][2]:.1f}] cm (动态)")
                            print(f"  角点2: [{current_corners['corner2'][0]:.1f}, {current_corners['corner2'][1]:.1f}, {current_corners['corner2'][2]:.1f}] cm (动态)")
                            print(f"  角点3: [{current_corners['corner3'][0]:.1f}, {current_corners['corner3'][1]:.1f}, {current_corners['corner3'][2]:.1f}] cm (动态)")
                            print(f"  角点4: [{current_corners['corner4'][0]:.1f}, {current_corners['corner4'][1]:.1f}, {current_corners['corner4'][2]:.1f}] cm (动态)")
                        else:
                            print(f"  角点1: [{STATIC_NEAR_LEFT[0]:.1f}, {STATIC_NEAR_LEFT[1]:.1f}, {STATIC_NEAR_LEFT[2]:.1f}] cm (静态)")
                            print(f"  角点2: [{STATIC_NEAR_RIGHT[0]:.1f}, {STATIC_NEAR_RIGHT[1]:.1f}, {STATIC_NEAR_RIGHT[2]:.1f}] cm (静态)")
                            print(f"  角点3: [{STATIC_FAR_RIGHT[0]:.1f}, {STATIC_FAR_RIGHT[1]:.1f}, {STATIC_FAR_RIGHT[2]:.1f}] cm (静态)")
                            print(f"  角点4: [{STATIC_FAR_LEFT[0]:.1f}, {STATIC_FAR_LEFT[1]:.1f}, {STATIC_FAR_LEFT[2]:.1f}] cm (静态)")
                    except Exception as send_err:
                        print(f"  [错误] 发送响应失败: {send_err}")
                else:
                    print("  [错误] 无法创建响应包")
                
                print("-" * 40)
                
            except socket.timeout:
                # 接收超时，正常现象
                continue
            except ConnectionResetError:
                print("[接收线程] 错误: 连接重置")
                time.sleep(0.5)
                continue
            except socket.error as sock_err:
                print(f"[接收线程] Socket错误: {sock_err}")
                # 尝试重新创建socket
                try:
                    print("[接收线程] 尝试重新创建接收socket...")
                    sock.close()
                    new_sock = create_udp_socket(bind_ip=local_ip, bind_port=local_port)
                    if new_sock:
                        sock = new_sock
                        print("[接收线程] 接收socket重新创建成功")
                    else:
                        print("[接收线程] 无法重新创建socket，线程退出")
                        break
                except Exception as recreate_err:
                    print(f"[接收线程] 重新创建socket失败: {recreate_err}")
                    break
            except Exception as e:
                print(f"[接收线程] 未知错误: {e}")
                time.sleep(0.5)
                continue
    finally:
        # 确保关闭socket
        try:
            if send_sock != sock and send_sock:
                send_sock.close()
                print("[接收线程] 发送socket已关闭")
        except Exception as e:
            print(f"[接收线程] 关闭socket时出错: {e}")
        print("[接收线程] 停止")

# --- 主动发送线程函数 ---
def sender_thread_func(local_ip, target_ip, target_port, interval=DEFAULT_SEND_INTERVAL):
    """主动发送数据包的线程"""
    print(f"[发送线程] 启动，从本地IP {local_ip} 向目标 {target_ip}:{target_port} 发送，间隔: {interval}秒")
    
    # 创建专用发送socket - 绑定到本地指定端口
    try:
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 设置地址重用选项，这样即使接收socket已经绑定了端口，发送socket也可以使用
        send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定到本地接口IP和指定端口
        try:
            # 这里我们尝试绑定到本地接口IP和端口
            send_sock.bind((local_ip, DEFAULT_LOCAL_PORT))
            print(f"[发送线程] 发送socket绑定到 {local_ip}:{DEFAULT_LOCAL_PORT} 成功")
        except Exception as bind_err:
            print(f"[发送线程] 绑定到 {local_ip}:{DEFAULT_LOCAL_PORT} 失败: {bind_err}")
            print("[发送线程] 尝试使用系统分配的随机端口...")
            # 如果绑定失败，继续使用随机端口
    except Exception as e:
        print(f"[发送线程] 创建socket失败: {e}")
        return
        
    # 获取并显示发送socket的本地地址
    try:
        local_addr = send_sock.getsockname()
        print(f"[发送线程] 本地发送地址: {local_addr[0]}:{local_addr[1]} (源端口)")
    except Exception as e:
        print(f"[发送线程] 无法获取本地端口信息: {e}")
    
    send_count = 0
    last_error_time = 0
    
    try:
        # 发送一个测试数据包
        try:
            test_data = b"HELLO_TEST_PACKET"
            dest_addr = (target_ip, target_port)
            print(f"[发送线程] 正在发送测试包到 {dest_addr[0]}:{dest_addr[1]}...")
            bytes_sent = send_sock.sendto(test_data, dest_addr)
            
            # 再次检查本地端口
            try:
                local_addr = send_sock.getsockname()
                print(f"[发送线程] 测试包已发送到 {dest_addr[0]}:{dest_addr[1]} (发送 {bytes_sent} 字节) [从本地地址: {local_addr[0]}:{local_addr[1]}]")
            except Exception as e:
                print(f"[发送线程] 测试包已发送到 {dest_addr[0]}:{dest_addr[1]} (发送 {bytes_sent} 字节)")
                print(f"[发送线程] 无法获取发送端口信息: {e}")
        except Exception as test_err:
            print(f"[发送线程] 测试包发送失败: {test_err}")
        
        # 主循环发送数据包
        while True:
            try:
                # 创建响应包
                packet = create_response_packet()
                if not packet:
                    print("[发送线程] 无法创建数据包，等待下一次发送...")
                    time.sleep(interval)
                    continue
                
                # 发送数据包
                dest_addr = (target_ip, target_port)
                print(f"[发送线程] 正在发送数据包 #{send_count+1} 到 {dest_addr[0]}:{dest_addr[1]}...")
                bytes_sent = send_sock.sendto(packet, dest_addr)
                
                # 获取本地端口信息
                try:
                    local_addr = send_sock.getsockname()
                    print(f"[主动发送] -> {dest_addr[0]}:{dest_addr[1]} (包 #{send_count+1}, {bytes_sent} 字节) [从本地地址: {local_addr[0]}:{local_addr[1]}]")
                except Exception as e:
                    print(f"[主动发送] -> {dest_addr[0]}:{dest_addr[1]} (包 #{send_count+1}, {bytes_sent} 字节)")
                    
                send_count += 1
                
                # 打印详细信息（仅每5个包显示一次）
                if send_count % 5 == 0:
                    print("  数据内容 (十六进制):")
                    print(hex_format(packet))
                
                # 等待指定间隔
                time.sleep(interval)
                
            except socket.error as sock_err:
                current_time = time.time()
                # 限制错误消息频率
                if current_time - last_error_time > 5:
                    print(f"[发送线程] Socket错误: {sock_err}")
                    last_error_time = current_time
                # 尝试重新创建socket
                try:
                    send_sock.close()
                    print("[发送线程] 尝试重新创建socket...")
                    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    
                    try:
                        send_sock.bind((local_ip, DEFAULT_LOCAL_PORT))
                        print(f"[发送线程] 发送socket重新绑定到 {local_ip}:{DEFAULT_LOCAL_PORT} 成功")
                    except Exception as bind_err:
                        print(f"[发送线程] 重新绑定失败: {bind_err}")
                        
                    print("[发送线程] socket重新创建成功")
                except Exception as recreate_err:
                    print(f"[发送线程] 重新创建socket失败: {recreate_err}")
                    break
                time.sleep(interval)
            except Exception as e:
                print(f"[发送线程] 未知错误: {e}")
                time.sleep(interval)
    finally:
        # 确保关闭socket
        try:
            if send_sock:
                send_sock.close()
                print("[发送线程] socket已关闭")
        except Exception as e:
            print(f"[发送线程] 关闭socket时出错: {e}")
        print("[发送线程] 停止")

# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser(description='探测子系统网络通信程序')
    parser.add_argument('--local_ip', type=str, default=DEFAULT_LOCAL_IP, 
                        help=f'本地监听IP (默认: {DEFAULT_LOCAL_IP})')
    parser.add_argument('--local_port', type=int, default=DEFAULT_LOCAL_PORT, 
                        help=f'本地监听端口 (默认: {DEFAULT_LOCAL_PORT})')
    parser.add_argument('--target_ip', type=str, default=DEFAULT_TARGET_IP, 
                        help=f'目标IP (默认: {DEFAULT_TARGET_IP})')
    parser.add_argument('--target_port', type=int, default=DEFAULT_TARGET_PORT, 
                        help=f'目标端口 (默认: {DEFAULT_TARGET_PORT})')
    parser.add_argument('--no-ros', action='store_true', 
                        help='无ROS模式：使用静态法向量')
    parser.add_argument('--active-send', action='store_true', 
                        help='启用主动发送模式')
    parser.add_argument('--send-interval', type=float, default=DEFAULT_SEND_INTERVAL, 
                        help=f'主动发送间隔(秒) (默认: {DEFAULT_SEND_INTERVAL})')
    args = parser.parse_args()

    # 显示网络接口信息
    print_network_interfaces()

    # 显示配置信息
    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"本机IP: {LOCAL_INTERFACE_IP}")
    print(f"监听设置: {args.local_ip}:{args.local_port}")
    print(f"目标IP: {args.target_ip}:{args.target_port}")
    print(f"ROS模式: {'禁用' if args.no_ros else '启用'}")
    print(f"主动发送: {'启用' if args.active_send else '禁用'}")
    if args.active_send:
        print(f"发送间隔: {args.send_interval}秒")
    print("数据格式: 小端序")
    print("=" * 60 + "\n")

    # 创建接收socket
    print(f"[主程序] 创建接收socket (监听IP: {args.local_ip})...")
    recv_sock = create_udp_socket(bind_ip=args.local_ip, bind_port=args.local_port)
    if not recv_sock:
        print("[错误] 无法创建接收socket，程序退出")
        return

    # 启动接收线程
    receiver = threading.Thread(target=receiver_thread_func, 
                               args=(recv_sock, args.target_ip, args.target_port), 
                               daemon=True)
    receiver.start()
    
    # 启动主动发送线程 (如果启用)
    if args.active_send:
        print(f"[主程序] 启动主动发送线程，目标: {args.target_ip}:{args.target_port}...")
        sender = threading.Thread(target=sender_thread_func,
                                 args=(args.local_ip, args.target_ip, args.target_port, args.send_interval),
                                 daemon=True)
        sender.start()
    else:
        print("[主程序] 主动发送模式未启用")

    # 初始化ROS (如果启用)
    ros_initialized = False
    if not args.no_ros and ROS_AVAILABLE:
        print("[主程序] 初始化ROS...")
        ros_initialized = setup_ros()
        if not ros_initialized:
            print("[警告] ROS初始化失败，使用静态角点数据")

    # 主循环
    try:
        if args.no_ros and not ROS_AVAILABLE:
            print("[主程序] 使用静态法向量。按Ctrl+C退出。")
            while True:
                time.sleep(1)
        else:
            print("[主程序] 进入ROS循环。按Ctrl+C退出。")
            rospy.spin()
    except KeyboardInterrupt:
        print("\n[主程序] 检测到Ctrl+C，正在停止...")
    except Exception as e:
        print(f"\n[主程序] 错误: {e}")
    finally:
        # 清理资源
        print("[主程序] 关闭socket...")
        if recv_sock and recv_sock.fileno() != -1:
            recv_sock.close()
        print("[主程序] 等待线程结束...")
        time.sleep(0.5)
        print("[主程序] 程序退出。")

if __name__ == "__main__":
    main() 

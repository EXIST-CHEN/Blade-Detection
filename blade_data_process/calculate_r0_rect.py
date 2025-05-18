import numpy as np
import cv2            
import math

# 已知相机内参矩阵 K
K = np.array([[505.603861, 0.000000, 312.618312],
              [0.000000, 505.845117, 239.323011],
              [0.000000, 0.000000, 1.000000]])

# 相机畸变
distortion = np.array([0.027965, -0.53150, -0.003981, -0.000803, 0.000000])

# 图像特征点（像素坐标），至少6组
image_points = np.array([
    [119, 90], # 左边x中心
    [196, 98], # 左边x右上角
    [335, 154], # 右边x左上角
    [358, 179], # 右边x中心
    [232, 433], # 后面台子底座的右上角
    [230, 457], # 后面台子底座的右下角
], dtype=np.float32)

# 点云特征点（3D坐标）
object_points = np.array([
    [9.425633, -9.153234, -3.024467], # 左边x中心
    [9.216154, -9.216154, -1.136947], # 左边x右上角
    [8.716186, -10.706161, 2.676096], # 右边x左上角
    [8.661661, -11.511125, 3.314396], # 右边x中心
    [1.972488, -11.686857, -0.617072], # 后面台子底座的右上角
    [1.568606, -11.803295, -0.619941], # 后面台子底座的右下角
], dtype=np.float32)


# 求解外参
def solveMatrix(flags, useRefineLM: bool):
    if flags == cv2.SOLVEPNP_P3P or flags == cv2.SOLVEPNP_AP3P:
        object_points_subset = object_points[:4]
        image_points_subset = image_points[:4]
    else:
        object_points_subset = object_points
        image_points_subset = image_points

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points_subset,
        imagePoints=image_points_subset,
        cameraMatrix=K,
        distCoeffs=distortion,
        flags=flags
    )

    if useRefineLM:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-8)
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=object_points_subset,
            imagePoints=image_points_subset,
            cameraMatrix=K,
            distCoeffs=distortion,
            rvec=rvec,
            tvec=tvec,
            criteria=criteria
        )

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(3, 1)

    # 设置科学计数法格式
    np.set_printoptions(
        formatter={'float_kind': lambda x: f"{x:.6e}"},  # 保留6位有效数字
        suppress=False,  # 强制科学计数法
        linewidth=1000   # 防止自动换行
    )

    print("Extrinsic Matrix (R | T):\n", np.hstack((R, T)), "\n")

    total_error_sum = 0.0
    max_error = 0.0
    for i in range(len(object_points_subset)):
        # 雷达坐标转相机坐标
        point_radar = object_points_subset[i].reshape(3, 1)
        point_camera = R @ point_radar + T

        # 投影到图像平面（包含畸变校正）
        projected, _ = cv2.projectPoints(point_radar, rvec, tvec, K, distortion)
        u, v = projected[0][0]

        # 计算总误差
        dx = image_points_subset[i][0] - u
        dy = image_points_subset[i][1] - v
        total_error = math.hypot(dx, dy)
        
        # 更新统计
        total_error_sum += total_error
        if total_error > max_error:
            max_error = total_error

        print(f"Point {i}:\tLidar Coordinates:({point_radar[0][0]:.6f},{point_radar[1][0]:.6f},{point_radar[2][0]:.6f});\tCamera Coordinates:({point_camera[0][0]:.6f},{point_camera[1][0]:.6f},{point_camera[2][0]:.6f});\tProjected to Image:({u:.2f},{v:.2f});\tExpected Pixel:({image_points_subset[i][0]:.0f},{image_points_subset[i][1]:.0f});\tPixel Error:({dx:.2f},{dy:.2f}), total={total_error:.2f}")

    # 输出统计结果
    average_error = total_error_sum / len(object_points_subset)
    print(f"\nAverage Pixel Error: {average_error:.2f} px")
    print(f"Maximum Pixel Error: {max_error:.2f} px")

if __name__ == "__main__":
    # 定义所有要测试的case
    cases = [
        # 增强型PnP
        {"flag": cv2.SOLVEPNP_EPNP, "use_refine": False, "description": "SOLVEPNP_EPNP without LM"},
        {"flag": cv2.SOLVEPNP_EPNP, "use_refine": True, "description": "SOLVEPNP_EPNP with LM"},
        # 迭代法（默认）
        {"flag": cv2.SOLVEPNP_ITERATIVE, "use_refine": False, "description": "SOLVEPNP_ITERATIVE without LM"},
        {"flag": cv2.SOLVEPNP_ITERATIVE, "use_refine": True, "description": "SOLVEPNP_ITERATIVE with LM"},
        # P3P算法
        {"flag": cv2.SOLVEPNP_P3P, "use_refine": False, "description": "SOLVEPNP_P3P without LM"},
        {"flag": cv2.SOLVEPNP_P3P, "use_refine": True, "description": "SOLVEPNP_P3P with LM"},
        # 直接线性变换
        {"flag": cv2.SOLVEPNP_DLS, "use_refine": False, "description": "SOLVEPNP_DLS without LM"},
        {"flag": cv2.SOLVEPNP_DLS, "use_refine": True, "description": "SOLVEPNP_DLS with LM"},
        # 改进UPnP
        {"flag": cv2.SOLVEPNP_UPNP, "use_refine": False, "description": "SOLVEPNP_UPNP without LM"},
        {"flag": cv2.SOLVEPNP_UPNP, "use_refine": True, "description": "SOLVEPNP_UPNP with LM"},
        # 自适应P3P
        {"flag": cv2.SOLVEPNP_AP3P, "use_refine": False, "description": "SOLVEPNP_AP3P without LM"},
        {"flag": cv2.SOLVEPNP_AP3P, "use_refine": True, "description": "SOLVEPNP_AP3P with LM"}
    ]

    for case in cases:
        print("\n" + "="*80)
        print(f"Testing Case: {case['description']}")
        print("="*80 + "\n")
        solveMatrix(case["flag"], case["use_refine"])

"""
================================================================================
Testing Case: SOLVEPNP_EPNP without LM
================================================================================

Extrinsic Matrix (R | T):
 [[-1.388382e-01 1.949930e-01 9.709283e-01 1.426896e+00]
 [-8.571211e-01 -5.147577e-01 -1.918469e-02 -2.169899e-01]
 [4.960519e-01 -8.348666e-01 2.386006e-01 6.153918e-02]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.603099,-3.526178,11.657232);     Projected to Image:(117.38,89.27);      Expected Pixel:(119,90);        Pixel Error:(1.62,0.73), total=1.78
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.753638,-3.350452,12.056213);     Projected to Image:(197.40,98.87);      Expected Pixel:(196,98);        Pixel Error:(-1.40,-0.87), total=1.64
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.727427,-2.228078,13.961954);      Projected to Image:(338.99,158.42);     Expected Pixel:(335,154);       Pixel Error:(-3.99,-4.42), total=5.95
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.197778,-1.779228,14.759243);      Projected to Image:(353.69,178.23);     Expected Pixel:(358,179);       Pixel Error:(4.31,0.77), total=4.38
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(-1.724949,4.120087,10.649729);      Projected to Image:(231.83,431.86);     Expected Pixel:(232,433);       Pixel Error:(0.17,1.14), total=1.15
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(-1.694365,4.526255,10.545908);      Projected to Image:(232.98,451.51);     Expected Pixel:(230,457);       Pixel Error:(-2.98,5.49), total=6.25

Average Pixel Error: 3.53 px
Maximum Pixel Error: 6.25 px

================================================================================
Testing Case: SOLVEPNP_EPNP with LM
================================================================================

Extrinsic Matrix (R | T):
 [[-1.397395e-01 1.587947e-01 9.773725e-01 1.057611e+00]
 [-8.245766e-01 -5.651492e-01 -2.607322e-02 -1.016234e+00]
 [5.482210e-01 -8.095620e-01 2.099121e-01 -2.244096e-01]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.669039,-3.536591,11.718159);     Projected to Image:(115.72,89.69);      Expected Pixel:(119,90);        Pixel Error:(3.28,0.31), total=3.29
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.804948,-3.377513,12.050469);     Projected to Image:(195.24,97.71);      Expected Pixel:(196,98);        Pixel Error:(0.76,0.29), total=0.82
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.755076,-2.222593,13.783032);      Projected to Image:(340.35,157.57);     Expected Pixel:(335,154);       Pixel Error:(-5.35,-3.57), total=6.43
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.258728,-1.739351,14.538796);      Projected to Image:(356.43,178.69);     Expected Pixel:(358,179);       Pixel Error:(1.57,0.31), total=1.60
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(-1.676944,3.978206,10.188654);      Projected to Image:(230.57,433.52);     Expected Pixel:(232,433);       Pixel Error:(1.43,-0.52), total=1.52
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(-1.641800,4.377117,10.060899);      Projected to Image:(231.83,454.13);     Expected Pixel:(230,457);       Pixel Error:(-1.83,2.87), total=3.40

Average Pixel Error: 2.84 px
Maximum Pixel Error: 6.43 px

================================================================================
Testing Case: SOLVEPNP_ITERATIVE without LM
================================================================================

Extrinsic Matrix (R | T):
 [[-2.371753e-01 8.288324e-01 -5.067393e-01 1.184003e+01]
 [9.432992e-01 3.211838e-01 8.383059e-02 -2.814846e+00]
 [2.322380e-01 -4.581242e-01 -8.580138e-01 -1.825141e+01]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(3.550620,2.882933,-9.274063);       Projected to Image:(123.12,84.97);      Expected Pixel:(119,90);        Pixel Error:(-4.12,5.03), total=6.50
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(2.591672,2.823355,-10.913405);      Projected to Image:(192.76,108.50);     Expected Pixel:(196,98);        Pixel Error:(3.24,-10.50), total=10.98
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(-0.456932,2.192818,-13.618552);     Projected to Image:(329.60,157.69);     Expected Pixel:(335,154);       Pixel Error:(5.40,-3.69), total=6.54
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(-1.434632,1.936353,-13.810111);     Projected to Image:(365.20,168.24);     Expected Pixel:(358,179);       Pixel Error:(-7.20,10.76), total=12.94
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(1.998451,-4.759558,-11.909830);     Projected to Image:(229.10,437.78);     Expected Pixel:(232,433);       Pixel Error:(2.90,-4.78), total=5.59
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(1.999188,-5.178178,-11.947822);     Projected to Image:(229.78,453.32);     Expected Pixel:(230,457);       Pixel Error:(0.22,3.68), total=3.69

Average Pixel Error: 7.71 px
Maximum Pixel Error: 12.94 px

================================================================================
Testing Case: SOLVEPNP_ITERATIVE with LM
================================================================================

Extrinsic Matrix (R | T):
 [[-2.371753e-01 8.288324e-01 -5.067393e-01 1.184003e+01]
 [9.432992e-01 3.211838e-01 8.383059e-02 -2.814846e+00]
 [2.322380e-01 -4.581242e-01 -8.580138e-01 -1.825141e+01]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(3.550620,2.882933,-9.274063);       Projected to Image:(123.12,84.97);      Expected Pixel:(119,90);        Pixel Error:(-4.12,5.03), total=6.50
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(2.591672,2.823355,-10.913405);      Projected to Image:(192.76,108.50);     Expected Pixel:(196,98);        Pixel Error:(3.24,-10.50), total=10.98
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(-0.456932,2.192818,-13.618552);     Projected to Image:(329.60,157.69);     Expected Pixel:(335,154);       Pixel Error:(5.40,-3.69), total=6.54
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(-1.434632,1.936353,-13.810111);     Projected to Image:(365.20,168.24);     Expected Pixel:(358,179);       Pixel Error:(-7.20,10.76), total=12.94
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(1.998451,-4.759558,-11.909830);     Projected to Image:(229.10,437.78);     Expected Pixel:(232,433);       Pixel Error:(2.90,-4.78), total=5.59
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(1.999188,-5.178178,-11.947822);     Projected to Image:(229.78,453.32);     Expected Pixel:(230,457);       Pixel Error:(0.22,3.68), total=3.69

Average Pixel Error: 7.71 px
Maximum Pixel Error: 12.94 px

================================================================================
Testing Case: SOLVEPNP_P3P without LM
================================================================================

Extrinsic Matrix (R | T):
 [[-6.559738e-01 -3.176062e-01 6.847078e-01 1.110419e+00]
 [3.950654e-01 -9.174460e-01 -4.707689e-02 -1.552024e+01]
 [6.431343e-01 2.396232e-01 7.272957e-01 9.159150e+00]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.236302,-3.256516,10.828089);     Projected to Image:(119.00,90.00);      Expected Pixel:(119,90);        Pixel Error:(-0.00,-0.00), total=0.00
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.786505,-3.370407,12.051073);     Projected to Image:(196.00,98.00);      Expected Pixel:(196,98);        Pixel Error:(0.00,0.00), total=0.00
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.625516,-2.380433,14.145696);      Projected to Image:(335.00,154.00);     Expected Pixel:(335,154);       Pixel Error:(0.00,0.00), total=0.00
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.353993,-1.693512,14.381975);      Projected to Image:(360.26,179.64);     Expected Pixel:(358,179);       Pixel Error:(-2.26,-0.64), total=2.35

Average Pixel Error: 0.59 px
Maximum Pixel Error: 2.35 px

================================================================================
Testing Case: SOLVEPNP_P3P with LM
================================================================================

Extrinsic Matrix (R | T):
 [[-6.571172e-01 -2.880914e-01 6.965632e-01 1.385059e+00]
 [3.495256e-01 -9.351882e-01 -5.705212e-02 -1.531175e+01]
 [6.678539e-01 2.059768e-01 7.152236e-01 8.699396e+00]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.278451,-3.284703,10.945818);     Projected to Image:(119.15,90.31);      Expected Pixel:(119,90);        Pixel Error:(-0.15,-0.31), total=0.34
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.807894,-3.406765,12.142955);     Projected to Image:(196.00,97.56);      Expected Pixel:(196,98);        Pixel Error:(0.00,0.44), total=0.44
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.605927,-2.405623,14.229321);      Projected to Image:(334.17,153.60);     Expected Pixel:(335,154);       Pixel Error:(0.83,0.40), total=0.92
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.318276,-1.708304,14.483630);      Projected to Image:(358.68,179.55);     Expected Pixel:(358,179);       Pixel Error:(-0.68,-0.55), total=0.87

Average Pixel Error: 0.64 px
Maximum Pixel Error: 0.92 px

================================================================================
Testing Case: SOLVEPNP_DLS without LM
================================================================================

Extrinsic Matrix (R | T):
 [[-1.388382e-01 1.949930e-01 9.709283e-01 1.426896e+00]
 [-8.571211e-01 -5.147577e-01 -1.918469e-02 -2.169899e-01]
 [4.960519e-01 -8.348666e-01 2.386006e-01 6.153918e-02]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.603099,-3.526178,11.657232);     Projected to Image:(117.38,89.27);      Expected Pixel:(119,90);        Pixel Error:(1.62,0.73), total=1.78
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.753638,-3.350452,12.056213);     Projected to Image:(197.40,98.87);      Expected Pixel:(196,98);        Pixel Error:(-1.40,-0.87), total=1.64
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.727427,-2.228078,13.961954);      Projected to Image:(338.99,158.42);     Expected Pixel:(335,154);       Pixel Error:(-3.99,-4.42), total=5.95
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.197778,-1.779228,14.759243);      Projected to Image:(353.69,178.23);     Expected Pixel:(358,179);       Pixel Error:(4.31,0.77), total=4.38
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(-1.724949,4.120087,10.649729);      Projected to Image:(231.83,431.86);     Expected Pixel:(232,433);       Pixel Error:(0.17,1.14), total=1.15
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(-1.694365,4.526255,10.545908);      Projected to Image:(232.98,451.51);     Expected Pixel:(230,457);       Pixel Error:(-2.98,5.49), total=6.25

Average Pixel Error: 3.53 px
Maximum Pixel Error: 6.25 px

================================================================================
Testing Case: SOLVEPNP_DLS with LM
================================================================================

Extrinsic Matrix (R | T):
 [[-1.397395e-01 1.587947e-01 9.773725e-01 1.057611e+00]
 [-8.245766e-01 -5.651492e-01 -2.607322e-02 -1.016234e+00]
 [5.482210e-01 -8.095620e-01 2.099121e-01 -2.244096e-01]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.669039,-3.536591,11.718159);     Projected to Image:(115.72,89.69);      Expected Pixel:(119,90);        Pixel Error:(3.28,0.31), total=3.29
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.804948,-3.377513,12.050469);     Projected to Image:(195.24,97.71);      Expected Pixel:(196,98);        Pixel Error:(0.76,0.29), total=0.82
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.755076,-2.222593,13.783032);      Projected to Image:(340.35,157.57);     Expected Pixel:(335,154);       Pixel Error:(-5.35,-3.57), total=6.43
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.258728,-1.739351,14.538796);      Projected to Image:(356.43,178.69);     Expected Pixel:(358,179);       Pixel Error:(1.57,0.31), total=1.60
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(-1.676944,3.978206,10.188654);      Projected to Image:(230.57,433.52);     Expected Pixel:(232,433);       Pixel Error:(1.43,-0.52), total=1.52
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(-1.641800,4.377117,10.060899);      Projected to Image:(231.83,454.13);     Expected Pixel:(230,457);       Pixel Error:(-1.83,2.87), total=3.40

Average Pixel Error: 2.84 px
Maximum Pixel Error: 6.43 px

================================================================================
Testing Case: SOLVEPNP_UPNP without LM
================================================================================

Extrinsic Matrix (R | T):
 [[-1.388382e-01 1.949930e-01 9.709283e-01 1.426896e+00]
 [-8.571211e-01 -5.147577e-01 -1.918469e-02 -2.169899e-01]
 [4.960519e-01 -8.348666e-01 2.386006e-01 6.153918e-02]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.603099,-3.526178,11.657232);     Projected to Image:(117.38,89.27);      Expected Pixel:(119,90);        Pixel Error:(1.62,0.73), total=1.78
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.753638,-3.350452,12.056213);     Projected to Image:(197.40,98.87);      Expected Pixel:(196,98);        Pixel Error:(-1.40,-0.87), total=1.64
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.727427,-2.228078,13.961954);      Projected to Image:(338.99,158.42);     Expected Pixel:(335,154);       Pixel Error:(-3.99,-4.42), total=5.95
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.197778,-1.779228,14.759243);      Projected to Image:(353.69,178.23);     Expected Pixel:(358,179);       Pixel Error:(4.31,0.77), total=4.38
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(-1.724949,4.120087,10.649729);      Projected to Image:(231.83,431.86);     Expected Pixel:(232,433);       Pixel Error:(0.17,1.14), total=1.15
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(-1.694365,4.526255,10.545908);      Projected to Image:(232.98,451.51);     Expected Pixel:(230,457);       Pixel Error:(-2.98,5.49), total=6.25

Average Pixel Error: 3.53 px
Maximum Pixel Error: 6.25 px

================================================================================
Testing Case: SOLVEPNP_UPNP with LM
================================================================================

Extrinsic Matrix (R | T):
 [[-1.397395e-01 1.587947e-01 9.773725e-01 1.057611e+00]
 [-8.245766e-01 -5.651492e-01 -2.607322e-02 -1.016234e+00]
 [5.482210e-01 -8.095620e-01 2.099121e-01 -2.244096e-01]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.669039,-3.536591,11.718159);     Projected to Image:(115.72,89.69);      Expected Pixel:(119,90);        Pixel Error:(3.28,0.31), total=3.29
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.804948,-3.377513,12.050469);     Projected to Image:(195.24,97.71);      Expected Pixel:(196,98);        Pixel Error:(0.76,0.29), total=0.82
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.755076,-2.222593,13.783032);      Projected to Image:(340.35,157.57);     Expected Pixel:(335,154);       Pixel Error:(-5.35,-3.57), total=6.43
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.258728,-1.739351,14.538796);      Projected to Image:(356.43,178.69);     Expected Pixel:(358,179);       Pixel Error:(1.57,0.31), total=1.60
Point 4:        Lidar Coordinates:(1.972488,-11.686857,-0.617072);      Camera Coordinates:(-1.676944,3.978206,10.188654);      Projected to Image:(230.57,433.52);     Expected Pixel:(232,433);       Pixel Error:(1.43,-0.52), total=1.52
Point 5:        Lidar Coordinates:(1.568606,-11.803295,-0.619941);      Camera Coordinates:(-1.641800,4.377117,10.060899);      Projected to Image:(231.83,454.13);     Expected Pixel:(230,457);       Pixel Error:(-1.83,2.87), total=3.40

Average Pixel Error: 2.84 px
Maximum Pixel Error: 6.43 px

================================================================================
Testing Case: SOLVEPNP_AP3P without LM
================================================================================

Extrinsic Matrix (R | T):
 [[-6.559738e-01 -3.176062e-01 6.847078e-01 1.110419e+00]
 [3.950654e-01 -9.174460e-01 -4.707689e-02 -1.552024e+01]
 [6.431343e-01 2.396232e-01 7.272957e-01 9.159150e+00]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.236302,-3.256516,10.828089);     Projected to Image:(119.00,90.00);      Expected Pixel:(119,90);        Pixel Error:(-0.00,-0.00), total=0.00
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.786505,-3.370407,12.051073);     Projected to Image:(196.00,98.00);      Expected Pixel:(196,98);        Pixel Error:(0.00,0.00), total=0.00
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.625516,-2.380433,14.145696);      Projected to Image:(335.00,154.00);     Expected Pixel:(335,154);       Pixel Error:(0.00,0.00), total=0.00
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.353993,-1.693512,14.381975);      Projected to Image:(360.26,179.64);     Expected Pixel:(358,179);       Pixel Error:(-2.26,-0.64), total=2.35

Average Pixel Error: 0.59 px
Maximum Pixel Error: 2.35 px

================================================================================
Testing Case: SOLVEPNP_AP3P with LM
================================================================================

Extrinsic Matrix (R | T):
 [[-6.571172e-01 -2.880914e-01 6.965632e-01 1.385059e+00]
 [3.495256e-01 -9.351882e-01 -5.705212e-02 -1.531175e+01]
 [6.678539e-01 2.059768e-01 7.152235e-01 8.699396e+00]] 

Point 0:        Lidar Coordinates:(9.425633,-9.153234,-3.024467);       Camera Coordinates:(-4.278451,-3.284703,10.945819);     Projected to Image:(119.15,90.31);      Expected Pixel:(119,90);        Pixel Error:(-0.15,-0.31), total=0.34
Point 1:        Lidar Coordinates:(9.216154,-9.216154,-1.136947);       Camera Coordinates:(-2.807894,-3.406765,12.142956);     Projected to Image:(196.00,97.56);      Expected Pixel:(196,98);        Pixel Error:(0.00,0.44), total=0.44
Point 2:        Lidar Coordinates:(8.716186,-10.706161,2.676096);       Camera Coordinates:(0.605927,-2.405623,14.229321);      Projected to Image:(334.17,153.60);     Expected Pixel:(335,154);       Pixel Error:(0.83,0.40), total=0.92
Point 3:        Lidar Coordinates:(8.661661,-11.511125,3.314396);       Camera Coordinates:(1.318276,-1.708304,14.483630);      Projected to Image:(358.68,179.55);     Expected Pixel:(358,179);       Pixel Error:(-0.68,-0.55), total=0.87

Average Pixel Error: 0.64 px
Maximum Pixel Error: 0.92 px
"""
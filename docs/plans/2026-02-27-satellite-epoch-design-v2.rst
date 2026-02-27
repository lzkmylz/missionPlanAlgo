================================================================================
卫星独立历元时间支持 - 设计文档 v2（修订版）
================================================================================

:日期: 2026-02-27
:修订: v2.0
:作者: Claude Code
:状态: 设计完成，待实施

================================================================================
修订记录
================================================================================

v2.0 (2026-02-27)
    - 修正默认历元处理逻辑：使用场景开始时间而非硬编码日期
    - 添加强制UTC时区处理，禁止naive datetime流转
    - 增加J2摄动修正（RAAN和近地点幅角长期项）
    - 改进API设计：TLE与epoch冲突时发出警告
    - 明确Java后端时间精度要求

================================================================================
1. 问题背景
================================================================================

1.1 当前问题
--------------------------------------------------------------------------------

在设置场景时，只设置了场景时间，卫星参数里没有设置轨道历元时间。
每颗卫星应该支持设置不同的轨道历元时间，不论是早于或晚于场景时间，
在进行规划时都能正确适配。

1.2 具体表现
--------------------------------------------------------------------------------

1. **简化轨道模型**: 硬编码使用 ``datetime(2024, 1, 1)`` 作为参考历元
2. **Java Orekit集成**: 错误地使用 ``start_time.isoformat()`` 作为历元
3. **Orbit类**: 缺少 ``epoch`` 字段存储历元时间
4. **场景文件**: 无法为每颗卫星指定不同的轨道历元

================================================================================
2. 设计目标
================================================================================

1. 支持每颗卫星设置独立的轨道历元时间
2. 历元可以早于、等于或晚于场景开始时间
3. **默认历元使用场景开始时间**（修正v1的错误）
4. **强制UTC时区处理**，禁止naive datetime流转
5. **考虑J2摄动**（RAAN和近地点幅角长期进动）
6. **API显式警告**：TLE与epoch冲突时提醒用户
7. 三种轨道配置方式都正确处理历元

================================================================================
3. 数据模型设计
================================================================================

3.1 Orbit类修改 (core/models/satellite.py)
--------------------------------------------------------------------------------

.. code-block:: python

    from datetime import datetime, timezone
    from typing import Optional
    from dataclasses import dataclass, field

    @dataclass
    class Orbit:
        """
        轨道参数

        支持三种配置方式：
        1. 轨道六根数
        2. TLE两行根数（历元从TLE自动解析）
        3. 简化参数

        优先级：TLE > 六根数 > 简化参数

        重要：所有datetime必须是UTC时区感知的（timezone-aware）
        """
        orbit_type: OrbitType = OrbitType.SSO

        # 轨道六根数
        semi_major_axis: Optional[float] = None
        eccentricity: float = 0.0
        inclination: float = 97.4
        raan: float = 0.0          # 升交点赤经（度）
        arg_of_perigee: float = 0.0  # 近地点幅角（度）
        mean_anomaly: float = 0.0    # 平近点角（度）

        # ★ 新增：轨道历元时间（必须是UTC timezone-aware）
        # TLE格式自动解析历元，不需要此字段
        epoch: Optional[datetime] = None

        # 简化参数
        altitude: float = 500000.0

        # TLE两行根数
        tle_line1: Optional[str] = None
        tle_line2: Optional[str] = None

        # 轨道数据来源标志
        source: OrbitSource = field(default=OrbitSource.SIMPLIFIED)

        def __post_init__(self):
            """初始化后处理：确保epoch带UTC时区"""
            if self.epoch is not None:
                # 强制转换为UTC时区感知
                if self.epoch.tzinfo is None:
                    # naive datetime假设为UTC
                    import warnings
                    warnings.warn(
                        f"Orbit epoch is naive datetime {self.epoch}, "
                        "assuming UTC. Please provide timezone-aware datetime.",
                        UserWarning
                    )
                    self.epoch = self.epoch.replace(tzinfo=timezone.utc)
                else:
                    # 转换为UTC
                    self.epoch = self.epoch.astimezone(timezone.utc)

    # ★ 新增：获取带时区感知的datetime工具函数
    def ensure_utc_datetime(dt: Optional[datetime]) -> Optional[datetime]:
        """确保datetime是UTC时区感知的"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

3.2 三种配置方式的历元处理（★修订）
--------------------------------------------------------------------------------

+----------------+------------------+-------------------------------------------+
| 配置方式       | 历元来源         | 处理逻辑                                  |
+================+==================+===========================================+
| TLE            | TLE第1行自动解析 | SGP4自动解析，如果用户同时提供epoch，     |
|                |                  | 发出**警告**"TLE内置历元将覆盖指定epoch"  |
+----------------+------------------+-------------------------------------------+
| 六根数         | orbit.epoch      | 如果为None，使用**scenario_start_time**   |
+----------------+------------------+-------------------------------------------+
| 简化参数       | orbit.epoch      | 如果为None，使用**scenario_start_time**   |
+----------------+------------------+-------------------------------------------+

3.3 TLE与Epoch冲突处理（★新增）
--------------------------------------------------------------------------------

.. code-block:: python

    import warnings

    def create_orbit_from_dict(orbit_data: dict) -> Orbit:
        # ... 解析其他参数 ...

        tle_line1 = orbit_data.get('tle_line1')
        tle_line2 = orbit_data.get('tle_line2')
        epoch_str = orbit_data.get('epoch')

        # TLE模式下epoch冲突检测
        if tle_line1 and tle_line2 and epoch_str:
            warnings.warn(
                f"Orbit configured with both TLE and explicit epoch ({epoch_str}). "
                f"TLE内置历元将覆盖指定epoch. "
                f"Remove 'epoch' field when using TLE to suppress this warning.",
                UserWarning,
                stacklevel=2
            )

        # ... 继续创建Orbit ...

================================================================================
4. 传播器修改方案（★重大修订）
================================================================================

4.1 修改文件清单
--------------------------------------------------------------------------------

1. ``core/models/satellite.py`` - Orbit类添加epoch字段、UTC强制转换、J2摄动
2. ``core/orbit/visibility/orekit_visibility.py`` - 简化模型（传入scenario_start_time）
3. ``core/orbit/visibility/stk_visibility.py`` - STK传播器
4. ``core/orbit/visibility/orekit_java_bridge.py`` - Java后端时区处理

4.2 ★修订：简化模型修改（传入scenario_start_time）
--------------------------------------------------------------------------------

.. code-block:: python

    def _propagate_simplified(
        self,
        satellite,
        dt: datetime,
        scenario_start_time: Optional[datetime] = None  # ★新增参数
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        简化的轨道传播模型（圆轨道近似，带J2摄动修正）

        Args:
            satellite: 卫星模型
            dt: 目标时间（必须是UTC timezone-aware）
            scenario_start_time: 场景开始时间，作为默认历元

        Returns:
            (position, velocity) in meters and m/s
        """
        import math
        from datetime import timezone

        # 确保dt是UTC时区感知的
        if dt.tzinfo is None:
            raise ValueError("dt must be timezone-aware (UTC)")
        dt = dt.astimezone(timezone.utc)

        # 获取轨道参数
        orbit = getattr(satellite, 'orbit', None)

        if orbit is None:
            altitude = 500000.0
            inclination = 97.4
            raan = 0.0
            arg_of_perigee = 0.0
            mean_anomaly_offset = 0.0
            # ★修正：使用scenario_start_time作为默认历元
            ref_time = ensure_utc_datetime(scenario_start_time) or datetime(2024, 1, 1, tzinfo=timezone.utc)
        else:
            altitude = getattr(orbit, 'altitude', 500000.0)
            inclination = getattr(orbit, 'inclination', 97.4)
            raan = getattr(orbit, 'raan', 0.0)
            arg_of_perigee = getattr(orbit, 'arg_of_perigee', 0.0)
            mean_anomaly_offset = getattr(orbit, 'mean_anomaly', 0.0)

            # ★修正：优先使用卫星epoch，否则使用scenario_start_time
            if getattr(orbit, 'epoch', None):
                ref_time = orbit.epoch  # 已经是UTC时区感知
            elif scenario_start_time:
                ref_time = ensure_utc_datetime(scenario_start_time)
            else:
                # 最后fallback到固定日期，但必须带时区
                ref_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # 确保ref_time也是UTC
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # 计算轨道参数
        r = self.EARTH_RADIUS + altitude
        GM = 3.986004418e14
        period = 2 * math.pi * math.sqrt(r**3 / GM)
        mean_motion = 2 * math.pi / period

        # 计算从历元到目标时间的偏移（秒）
        delta_t = (dt - ref_time).total_seconds()

        # ★新增：J2摄动修正（长期项）
        # 地球J2项系数
        J2 = 1.08263e-3
        R_earth = 6371000.0  # 地球半径（米）

        # 半长轴和偏心率
        a = r
        e = getattr(orbit, 'eccentricity', 0.0) if orbit else 0.0

        # 计算J2摄动引起的RAAN进动速率（rad/s）
        # dRAAN/dt = -3/2 * n * J2 * (R/a)^2 * cos(i) / (1-e^2)^2
        n = mean_motion  # 平均运动（rad/s）
        cos_i = math.cos(math.radians(inclination))
        factor = (1 - e**2)**2

        if factor > 0:
            raan_dot = -1.5 * n * J2 * (R_earth / a)**2 * cos_i / factor
        else:
            raan_dot = 0.0

        # 计算J2摄动引起的近地点幅角变化速率（rad/s）
        # dω/dt = 3/4 * n * J2 * (R/a)^2 * (5*cos^2(i) - 1) / (1-e^2)^2
        cos_i_sq = cos_i ** 2
        if factor > 0:
            arg_perigee_dot = 0.75 * n * J2 * (R_earth / a)**2 * (5*cos_i_sq - 1) / factor
        else:
            arg_perigee_dot = 0.0

        # 应用J2摄动修正后的轨道参数
        raan_corrected = raan + math.degrees(raan_dot * delta_t)
        arg_perigee_corrected = arg_of_perigee + math.degrees(arg_perigee_dot * delta_t)

        # 平近点角（考虑初始mean_anomaly和历元偏移）
        mean_anomaly = math.radians(mean_anomaly_offset) + mean_motion * delta_t

        # 轨道参数（使用修正后的RAAN和近地点幅角）
        i = math.radians(inclination)
        raan_rad = math.radians(raan_corrected)

        # 对于圆轨道（e≈0），近地点幅角不重要，但保留计算
        # 对于小偏心率轨道，可能需要考虑

        # 圆轨道位置（在轨道平面内）
        x_orb = r * math.cos(mean_anomaly)
        y_orb = r * math.sin(mean_anomaly)

        # 转换到ECI坐标系（使用修正后的RAAN）
        x_eci = x_orb * math.cos(raan_rad) - y_orb * math.cos(i) * math.sin(raan_rad)
        y_eci = x_orb * math.sin(raan_rad) + y_orb * math.cos(i) * math.cos(raan_rad)
        z_eci = y_orb * math.sin(i)

        # 地球自转角速度 (rad/s)
        omega_earth = 7.2921159e-5

        # 初始偏移量（简化模型与Orekit对齐）
        theta_0 = math.radians(100.0)
        theta = theta_0 + omega_earth * (dt - datetime(2024, 1, 1, tzinfo=timezone.utc)).total_seconds()

        # 将ECI坐标转换为ECEF坐标（考虑地球自转）
        x = x_eci * math.cos(theta) + y_eci * math.sin(theta)
        y = -x_eci * math.sin(theta) + y_eci * math.cos(theta)
        z = z_eci

        # 计算速度
        v = math.sqrt(GM / r)
        vx_orb = -v * math.sin(mean_anomaly)
        vy_orb = v * math.cos(mean_anomaly)

        vx_eci = vx_orb * math.cos(raan_rad) - vy_orb * math.cos(i) * math.sin(raan_rad)
        vy_eci = vx_orb * math.sin(raan_rad) + vy_orb * math.cos(i) * math.cos(raan_rad)
        vz_eci = vy_orb * math.sin(i)

        # 转换速度到ECEF坐标系
        vx = vx_eci * math.cos(theta) + vy_eci * math.sin(theta) - omega_earth * y
        vy = -vx_eci * math.sin(theta) + vy_eci * math.cos(theta) + omega_earth * x
        vz = vz_eci

        return ((x, y, z), (vx, vy, vz))

4.3 ★修订：Java Orekit传播修改（UTC时区强制）
--------------------------------------------------------------------------------

.. code-block:: python

    def _propagate_range_with_java_orekit(
        self,
        satellite,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta
    ) -> List[Tuple[...]]:

        from datetime import timezone

        # ★确保所有时间都是UTC时区感知的
        if start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware (UTC)")
        if end_time.tzinfo is None:
            raise ValueError("end_time must be timezone-aware (UTC)")

        start_time = start_time.astimezone(timezone.utc)
        end_time = end_time.astimezone(timezone.utc)

        # ...

        # 获取卫星指定的历元
        orbit = getattr(satellite, 'orbit', None)
        sat_epoch = getattr(orbit, 'epoch', None) if orbit else None

        # ★确保sat_epoch也是UTC时区感知的
        if sat_epoch and sat_epoch.tzinfo is None:
            sat_epoch = sat_epoch.replace(tzinfo=timezone.utc)

        # 使用卫星指定的历元创建轨道
        if sat_epoch:
            epoch_date = AbsoluteDate(
                sat_epoch.year, sat_epoch.month, sat_epoch.day,
                sat_epoch.hour, sat_epoch.minute,
                sat_epoch.second + sat_epoch.microsecond / 1e6,
                utc
            )
        else:
            # 默认使用传播开始时间（向后兼容）
            epoch_date = AbsoluteDate(
                start_time.year, start_time.month, start_time.day,
                start_time.hour, start_time.minute,
                start_time.second + start_time.microsecond / 1e6,
                utc
            )

        # ... 后续代码 ...

4.4 ★修订：Java后端修改（高精度时间格式）
--------------------------------------------------------------------------------

.. code-block:: python

    for sat in satellites:
        sat_param = SatelliteParameters()
        sat_param.setId(sat['id'])
        sat_param.setSemiMajorAxis(sat.get('semiMajorAxis', 7016000.0))
        sat_param.setEccentricity(sat.get('eccentricity', 0.001))
        sat_param.setInclination(sat.get('inclination', 97.9))
        sat_param.setRaan(sat.get('raan', 0.0))
        sat_param.setArgOfPerigee(sat.get('argOfPerigee', 90.0))
        sat_param.setMeanAnomaly(sat.get('meanAnomaly', 0.0))
        sat_param.setAltitude(sat.get('altitude', 645000.0))

        # ★使用高精度ISO格式（保留微秒）并明确UTC
        sat_epoch = sat.get('epoch')
        if sat_epoch:
            # 确保格式包含微秒和UTC时区
            if isinstance(sat_epoch, datetime):
                # Python datetime转ISO格式（保留微秒）
                epoch_str = sat_epoch.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            else:
                epoch_str = sat_epoch
            sat_param.setEpoch(epoch_str)
        else:
            # 使用场景开始时间，同样高精度格式
            epoch_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            sat_param.setEpoch(epoch_str)

        sat_list.add(sat_param)

4.5 传播接口修改（传入scenario_start_time）
--------------------------------------------------------------------------------

所有传播方法都需要修改签名，传入 ``scenario_start_time``：

.. code-block:: python

    class OrekitVisibilityCalculator:

        def compute_satellite_target_windows(
            self,
            satellite,
            target,
            start_time: datetime,
            end_time: datetime,
            scenario_start_time: Optional[datetime] = None,  # ★新增
            time_step: Optional[timedelta] = None
        ) -> List[VisibilityWindow]:
            # 传播时传入scenario_start_time
            sat_positions = self._propagate_range(
                satellite, start_time, end_time, time_step,
                scenario_start_time=scenario_start_time  # ★传入
            )

        def _propagate_range(
            self,
            satellite,
            start_time: datetime,
            end_time: datetime,
            time_step: timedelta,
            scenario_start_time: Optional[datetime] = None  # ★新增
        ):
            # 调用简化模型时传入
            pos, vel = self._propagate_simplified(
                satellite, current_time,
                scenario_start_time=scenario_start_time  # ★传入
            )

================================================================================
5. 序列化/反序列化支持（★修订：强制UTC）
================================================================================

5.1 to_dict方法（带UTC标记）
--------------------------------------------------------------------------------

.. code-block:: python

    def to_dict(self) -> Dict[str, Any]:
        orbit_dict: Dict[str, Any] = {
            'orbit_type': self.orbit.orbit_type.value,
            'source': self.orbit.source.value,
        }

        # ★添加epoch字段（ISO格式带Z表示UTC）
        if self.orbit.epoch:
            # 确保输出UTC时间并带Z后缀
            epoch_utc = self.orbit.epoch.astimezone(timezone.utc)
            orbit_dict['epoch'] = epoch_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        # ... 其他字段 ...

5.2 from_dict方法（强制UTC解析）
--------------------------------------------------------------------------------

.. code-block:: python

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Satellite':
        # ...
        orbit_data = data.get('orbit', {})

        # 解析epoch字段（强制UTC）
        epoch_str = orbit_data.get('epoch')
        epoch = None
        if epoch_str:
            # 解析多种格式，统一转换为UTC timezone-aware
            epoch = parse_epoch_string(epoch_str)

        # ...

    def parse_epoch_string(epoch_str: str) -> datetime:
        """解析epoch字符串，返回UTC timezone-aware datetime"""
        from datetime import datetime, timezone

        # 尝试ISO 8601格式（带时区）
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f%z',  # 带微秒和时区
            '%Y-%m-%dT%H:%M:%S%z',      # 带时区
            '%Y-%m-%dT%H:%M:%S.%fZ',    # 带微秒和Z
            '%Y-%m-%dT%H:%M:%SZ',       # 带Z
            '%Y-%m-%dT%H:%M:%S.%f',     # 带微秒无TZ
            '%Y-%m-%dT%H:%M:%S',        # 无TZ
            '%Y-%m-%d %H:%M:%S',        # 空格分隔
            '%Y-%m-%d',                 # 日期only
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(epoch_str, fmt)
                # 如果没有时区信息，假设为UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    # 转换为UTC
                    dt = dt.astimezone(timezone.utc)
                return dt
            except ValueError:
                continue

        raise ValueError(f"Cannot parse epoch string: {epoch_str}")

================================================================================
6. 测试策略（★新增：时区和J2摄动测试）
================================================================================

6.1 单元测试
--------------------------------------------------------------------------------

1. **Orbit类测试**:
   - epoch字段序列化/反序列化
   - **★naive datetime自动转换UTC警告**
   - **★带时区datetime正确转换为UTC**

2. **传播器测试**:
   - 不同历元的卫星传播结果正确
   - **★J2摄动修正效果验证**（RAAN进动率是否符合理论值）
   - **★naive datetime传入时抛出ValueError**

3. **边界测试**:
   - 历元早于/等于/晚于场景时间
   - **★跨时区场景测试**（历元在UTC+8，场景在UTC）

6.2 集成测试
--------------------------------------------------------------------------------

1. **场景加载测试**: 从JSON加载带历元的卫星配置
2. **可见性计算测试**: 不同历元的卫星可见窗口计算正确
3. **多历元卫星测试**: 同一场景中包含多个不同历元的卫星
4. **★J2摄动长期传播测试**: 历元与场景时间相差数周时的精度

6.3 向后兼容测试
--------------------------------------------------------------------------------

1. **无epoch场景**: 旧场景文件仍能正常工作
2. **默认历元**: 未设置epoch时使用scenario_start_time
3. **★TLE+epoch冲突警告**: 验证警告是否正确发出

================================================================================
7. 实施步骤（★修订）
================================================================================

Phase 1: 数据模型和工具函数
    - 修改Orbit类添加epoch字段
    - ★添加ensure_utc_datetime工具函数
    - ★添加parse_epoch_string工具函数
    - 修改to_dict和from_dict方法（强制UTC）
    - ★添加TLE与epoch冲突警告
    - 运行单元测试

Phase 2: 传播器接口修改
    - 修改所有传播方法签名，添加scenario_start_time参数
    - 更新调用链传递scenario_start_time
    - ★修改简化模型：默认历元使用scenario_start_time
    - ★在简化模型中实现J2摄动修正
    - 运行单元测试

Phase 3: 具体传播器实现
    - 修改orekit_visibility.py的简化模型
    - 修改orekit_visibility.py的Java Orekit传播（UTC强制）
    - 修改stk_visibility.py的STK传播器
    - 修改orekit_java_bridge.py的Java后端（高精度时间格式）
    - 运行单元测试

Phase 4: 集成测试
    - 创建多历元测试场景（包含J2摄动验证）
    - 验证时区处理正确
    - 验证TLE+epoch冲突警告
    - 运行向后兼容测试

Phase 5: 文档更新
    - 更新场景文件格式文档（强调UTC时区）
    - 添加使用示例
    - 更新API文档

================================================================================
8. 风险评估（★更新）
================================================================================

+----------------------+--------+-------------------------------------------+
| 风险                 | 影响   | 缓解措施                                  |
+======================+========+===========================================+
| **时区处理错误**     | **高** | 强制UTC转换，naive datetime抛出异常或警告 |
+----------------------+--------+-------------------------------------------+
| J2摄动计算复杂度     | 中     | 提供开关允许用户禁用J2修正（简化模式）    |
+----------------------+--------+-------------------------------------------+
| 接口破坏性变更       | 中     | scenario_start_time使用Optional默认None   |
+----------------------+--------+-------------------------------------------+
| 向后兼容性问题       | 高     | 保持默认行为，epoch为None时用场景开始时间 |
+----------------------+--------+-------------------------------------------+
| 时间解析错误         | 中     | 支持多种格式，添加错误处理和日志          |
+----------------------+--------+-------------------------------------------+

================================================================================
9. 附录
================================================================================

9.1 时间格式规范（★严格）
--------------------------------------------------------------------------------

系统**只接受**以下格式的epoch字符串，**全部解释为UTC**：

- 推荐格式（带Z后缀）: ``"2024-01-01T00:00:00Z"``
- 带微秒: ``"2024-01-01T00:00:00.123456Z"``
- ISO 8601带时区: ``"2024-01-01T00:00:00+00:00"``

**警告**: 以下格式会被解析但发出警告（假设为UTC）：
- 无后缀: ``"2024-01-01T00:00:00"``
- 日期only: ``"2024-01-01"``

9.2 J2摄动参数
--------------------------------------------------------------------------------

- 地球J2项: 1.08263e-3
- 地球半径: 6371000.0 m
- RAAN进动公式: dRAAN/dt = -3/2 * n * J2 * (R/a)^2 * cos(i) / (1-e^2)^2
- 近地点幅角变化: dω/dt = 3/4 * n * J2 * (R/a)^2 * (5*cos^2(i) - 1) / (1-e^2)^2

9.3 相关文件
--------------------------------------------------------------------------------

- core/models/satellite.py
- core/orbit/visibility/orekit_visibility.py
- core/orbit/visibility/stk_visibility.py
- core/orbit/visibility/orekit_java_bridge.py
- java/src/orekit/visibility/SatelliteParameters.java

================================================================================
文档版本: 2.0
最后更新: 2026-02-27
修订说明: 修正默认历元逻辑，强制UTC时区，添加J2摄动，改进API警告
================================================================================

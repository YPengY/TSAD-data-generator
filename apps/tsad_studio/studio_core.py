from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthtsad.anomaly.local import AnomalyEvent, LocalAnomalyInjector
from synthtsad.anomaly.seasonal import SeasonalAnomalyInjector
from synthtsad.causal.arx import ARXSystem
from synthtsad.causal.dag import CausalGraphSampler
from synthtsad.components.noise import render_noise
from synthtsad.components.seasonality import render_seasonality
from synthtsad.components.trend import render_trend
from synthtsad.config import (
    DEFAULT_CONFIG,
    LOCAL_NODE_POLICY_MODES,
    LOCAL_TARGET_COMPONENTS,
    SEASONAL_NODE_POLICY_MODES,
    SEASONAL_TARGET_COMPONENTS,
    load_config_from_raw,
)
from synthtsad.interfaces import GenerationMetadata, LabelPayload, Stage1NodeParams
from synthtsad.labeling.labeler import LabelBuilder
from synthtsad.pipeline import SyntheticGeneratorPipeline


CONFIG_PATH = REPO_ROOT / "configs" / "default.json"

SECTION_LABELS = {
    "root": "Project Config",
    "num_samples": "Number of Samples",
    "sequence_length": "Sequence Length",
    "anomaly_sample_ratio": "Anomalous Sample Ratio",
    "num_series": "Number of Series",
    "seed": "Global Seed",
    "weights": "Sampling Weights",
    "stage1": "Stage 1 Baseline",
    "stage1.trend": "Trend",
    "stage1.seasonality": "Seasonality",
    "stage1.noise": "Noise",
    "causal": "Stage 2 Causal",
    "anomaly": "Stage 3 Anomalies",
    "anomaly.defaults": "Placement Policy",
    "anomaly.local": "Local Operators",
    "anomaly.local.budget": "Local Budget",
    "anomaly.local.defaults": "Local Defaults",
    "anomaly.local.type_weights": "Local Type Weights",
    "anomaly.local.per_type": "Local Per-Type Specs",
    "anomaly.seasonal": "Seasonal Operators",
    "anomaly.seasonal.budget": "Seasonal Budget",
    "anomaly.seasonal.defaults": "Seasonal Defaults",
    "anomaly.seasonal.type_weights": "Seasonal Type Weights",
    "anomaly.seasonal.per_type": "Seasonal Per-Type Specs",
    "debug": "Debug Toggles",
}

SECTION_LABELS_ZH = {
    "root": "项目配置",
    "num_samples": "样本数量",
    "sequence_length": "序列长度",
    "anomaly_sample_ratio": "异常样本比例",
    "num_series": "序列条数",
    "seed": "全局随机种子",
    "weights": "采样权重",
    "stage1": "阶段 1 基线",
    "stage1.trend": "趋势组件",
    "stage1.seasonality": "季节组件",
    "stage1.noise": "噪声组件",
    "causal": "阶段 2 因果结构",
    "anomaly": "阶段 3 异常",
    "anomaly.defaults": "放置策略",
    "anomaly.local": "局部异常算子",
    "anomaly.local.budget": "局部事件预算",
    "anomaly.local.defaults": "局部默认策略",
    "anomaly.local.type_weights": "局部类型权重",
    "anomaly.local.per_type": "局部逐类型参数",
    "anomaly.seasonal": "季节异常算子",
    "anomaly.seasonal.budget": "季节事件预算",
    "anomaly.seasonal.defaults": "季节默认策略",
    "anomaly.seasonal.type_weights": "季节类型权重",
    "anomaly.seasonal.per_type": "季节逐类型参数",
    "debug": "调试开关",
}

FIELD_LABELS = {
    "min": "Min",
    "max": "Max",
    "num_samples": "Sample Count",
    "sequence_length": "Sequence Length",
    "anomaly_sample_ratio": "Anomalous Sample Ratio",
    "num_series": "Series Count",
    "seed": "Seed",
    "change_points": "Change Points",
    "slope_scale": "Slope Scale",
    "arima_noise_scale": "ARIMA Noise Scale",
    "arima": "ARIMA Settings",
    "p_max": "AR Order Max",
    "q_max": "MA Order Max",
    "d": "Difference Order",
    "coef_bound": "Coefficient Bound",
    "atoms": "Seasonal Atoms",
    "amplitude": "Amplitude Range",
    "base_period": "Base Period",
    "low": "Low Frequency Period",
    "high": "High Frequency Period",
    "wavelet": "Wavelet Settings",
    "families": "Wavelet Family Weights",
    "scale": "Scale Range",
    "shift": "Shift Range",
    "contrastive": "Contrastive Pairing",
    "ratio": "Contrastive Ratio",
    "params": "Contrastive Params",
    "sigma": "Noise Sigma",
    "volatility_windows": "Volatility Windows",
    "volatility_multiplier": "Volatility Multiplier",
    "num_nodes": "Causal Node Range",
    "edge_density": "Edge Density",
    "max_lag": "Max Lag",
    "a_i_bound": "Self Dynamic Bound",
    "bias_std": "Bias Std",
    "b_ij_std": "Edge Gain Std",
    "alpha_i_min": "Alpha Min",
    "alpha_i_max": "Alpha Max",
    "defaults": "Defaults",
    "local": "Local Family",
    "seasonal": "Seasonal Family",
    "budget": "Budget",
    "allow_overlap": "Allow Overlap",
    "min_gap": "Minimum Gap",
    "max_events_per_node": "Max Events Per Node",
    "events_per_sample": "Events Per Sample",
    "activation_p": "Activation Probability",
    "window_length": "Event Window Length",
    "type_weights": "Type Weights",
    "per_type": "Per-Type Specs",
    "enabled": "Enabled",
    "endogenous_p": "Endogenous Probability",
    "target_component": "Target Component",
    "node_policy": "Node Policy",
    "mode": "Sampling Mode",
    "applies_to": "Applies To",
    "half_width": "Half Width",
    "spike_count": "Spike Count",
    "stride": "Stride",
    "rise_length": "Rise Length",
    "fall_length": "Fall Length",
    "kappa": "Sigmoid Slope",
    "rapid_tau": "Rapid Time Constant",
    "slow_tau": "Slow Time Constant",
    "shift_magnitude": "Shift Magnitude",
    "frequency": "Frequency",
    "positive_p": "Positive Sign Probability",
    "min_abs_amplitude": "Minimum Absolute Amplitude",
    "scale": "Scale Range",
    "factor": "Scaling Factor",
    "delta_phase": "Phase Shift",
    "amplitude_scale": "Amplitude Scale",
    "depth": "Modulation Depth",
    "delta_cycle": "Cycle Shift",
    "target_type_weights": "Target Type Weights",
    "target_family_weights": "Target Family Weights",
    "family_weights": "Family Weights",
    "order": "Harmonic Order",
    "noise_scale": "Noise Scale",
    "enable_trend": "Enable Trend",
    "enable_seasonality": "Enable Seasonality",
    "enable_noise": "Enable Noise",
    "enable_causal": "Enable Causal",
    "enable_local_anomaly": "Enable Local Anomaly",
    "enable_seasonal_anomaly": "Enable Seasonal Anomaly",
}

FIELD_LABELS_ZH = {
    "min": "最小值",
    "max": "最大值",
    "num_samples": "样本数",
    "sequence_length": "序列长度",
    "anomaly_sample_ratio": "异常样本比例",
    "num_series": "序列条数",
    "seed": "随机种子",
    "change_points": "变化点数量",
    "slope_scale": "斜率尺度",
    "arima_noise_scale": "ARIMA 噪声尺度",
    "arima": "ARIMA 设置",
    "p_max": "AR 阶数上限",
    "q_max": "MA 阶数上限",
    "d": "差分阶数",
    "coef_bound": "系数范围",
    "atoms": "季节原子数",
    "amplitude": "振幅范围",
    "base_period": "基础周期",
    "low": "低频周期",
    "high": "高频周期",
    "wavelet": "小波设置",
    "families": "小波族权重",
    "scale": "尺度范围",
    "shift": "平移范围",
    "contrastive": "对比配对",
    "ratio": "配对比例",
    "params": "对比参数",
    "sigma": "噪声强度",
    "volatility_windows": "波动率窗口",
    "volatility_multiplier": "波动率放大范围",
    "num_nodes": "因果节点范围",
    "edge_density": "边密度",
    "max_lag": "最大滞后",
    "a_i_bound": "自动态范围",
    "bias_std": "偏置标准差",
    "b_ij_std": "边增益标准差",
    "alpha_i_min": "Alpha 最小值",
    "alpha_i_max": "Alpha 最大值",
    "defaults": "默认项",
    "local": "局部异常族",
    "seasonal": "季节异常族",
    "budget": "预算",
    "allow_overlap": "允许重叠",
    "min_gap": "最小间隔",
    "max_events_per_node": "单节点最大事件数",
    "events_per_sample": "每样本事件数",
    "activation_p": "激活概率",
    "window_length": "事件窗口长度",
    "type_weights": "类型权重",
    "per_type": "逐类型参数",
    "enabled": "启用",
    "endogenous_p": "内生概率",
    "target_component": "目标组件",
    "node_policy": "节点策略",
    "mode": "采样模式",
    "applies_to": "适用波形",
    "half_width": "半宽",
    "spike_count": "尖峰数量",
    "stride": "间隔步长",
    "rise_length": "上升长度",
    "fall_length": "下降长度",
    "kappa": "Sigmoid 斜率",
    "rapid_tau": "快时常数",
    "slow_tau": "慢时常数",
    "shift_magnitude": "位移幅度",
    "frequency": "频率",
    "positive_p": "正号概率",
    "min_abs_amplitude": "最小绝对幅值",
    "factor": "缩放因子",
    "delta_phase": "相位偏移",
    "amplitude_scale": "幅值比例",
    "depth": "调制深度",
    "delta_cycle": "周期平移",
    "target_type_weights": "目标类型权重",
    "target_family_weights": "目标小波族权重",
    "family_weights": "小波族权重",
    "order": "谐波阶数",
    "noise_scale": "噪声尺度",
    "enable_trend": "启用趋势",
    "enable_seasonality": "启用季节性",
    "enable_noise": "启用噪声",
    "enable_causal": "启用因果",
    "enable_local_anomaly": "启用局部异常",
    "enable_seasonal_anomaly": "启用季节异常",
}

PATH_LABEL_OVERRIDES = {
    "weights.seasonality_type": "Seasonality Type Weights",
    "weights.seasonality_type.none": "None Weight",
    "weights.seasonality_type.sine": "Sine Weight",
    "weights.seasonality_type.square": "Square Weight",
    "weights.seasonality_type.triangle": "Triangle Weight",
    "weights.seasonality_type.wavelet": "Wavelet Weight",
    "weights.trend_type": "Trend Type Weights",
    "weights.trend_type.decrease": "Decrease Weight",
    "weights.trend_type.increase": "Increase Weight",
    "weights.trend_type.keep_steady": "Keep Steady Weight",
    "weights.trend_type.multiple": "Multiple Weight",
    "weights.trend_type.arima": "ARIMA Weight",
    "weights.frequency_regime": "Frequency Regime Weights",
    "weights.frequency_regime.low": "Low Regime Weight",
    "weights.frequency_regime.high": "High Regime Weight",
    "weights.noise_level": "Noise Level Weights",
    "weights.noise_level.almost_none": "Almost None Weight",
    "weights.noise_level.low": "Low Noise Weight",
    "weights.noise_level.moderate": "Moderate Noise Weight",
    "weights.noise_level.high": "High Noise Weight",
    "stage1.noise.sigma.almost_none": "Almost None Sigma",
    "stage1.noise.sigma.low": "Low Noise Sigma",
    "stage1.noise.sigma.moderate": "Moderate Noise Sigma",
    "stage1.noise.sigma.high": "High Noise Sigma",
}

PATH_LABEL_OVERRIDES_ZH = {
    "weights.seasonality_type": "季节类型权重",
    "weights.seasonality_type.none": "无季节权重",
    "weights.seasonality_type.sine": "正弦权重",
    "weights.seasonality_type.square": "方波权重",
    "weights.seasonality_type.triangle": "三角波权重",
    "weights.seasonality_type.wavelet": "小波权重",
    "weights.trend_type": "趋势类型权重",
    "weights.trend_type.decrease": "下降趋势权重",
    "weights.trend_type.increase": "上升趋势权重",
    "weights.trend_type.keep_steady": "平稳趋势权重",
    "weights.trend_type.multiple": "多段趋势权重",
    "weights.trend_type.arima": "ARIMA 权重",
    "weights.frequency_regime": "频率区间权重",
    "weights.frequency_regime.low": "低频权重",
    "weights.frequency_regime.high": "高频权重",
    "weights.noise_level": "噪声等级权重",
    "weights.noise_level.almost_none": "几乎无噪声权重",
    "weights.noise_level.low": "低噪声权重",
    "weights.noise_level.moderate": "中噪声权重",
    "weights.noise_level.high": "高噪声权重",
    "stage1.noise.sigma.almost_none": "几乎无噪声强度",
    "stage1.noise.sigma.low": "低噪声强度",
    "stage1.noise.sigma.moderate": "中噪声强度",
    "stage1.noise.sigma.high": "高噪声强度",
}

SECTION_DESCRIPTIONS = {
    "root": "Top-level generator defaults and stage switches.",
    "weights": "Sampling probabilities that control which baseline families are chosen.",
    "stage1": "Controls the baseline time-series components before causal mixing or anomaly injection.",
    "stage1.trend": "Defines how slow-moving trend shapes are sampled and rendered.",
    "stage1.seasonality": "Defines periodic structure, atoms, and wavelet-specific settings.",
    "stage1.noise": "Controls stochastic noise scale and volatility bursts.",
    "causal": "Controls DAG sparsity and ARX dynamics for cross-series propagation.",
    "anomaly": "Controls how many anomaly events are sampled and what archetypes are allowed.",
    "debug": "Turns individual generator stages on or off without changing the config structure.",
}

SECTION_DESCRIPTIONS_ZH = {
    "root": "整套数据生成流程的总开关和默认参数。",
    "weights": "控制各类基线模式被采样到的概率。",
    "stage1": "定义因果混合和异常注入之前的基础时间序列组件。",
    "stage1.trend": "定义慢变化趋势的采样和渲染方式。",
    "stage1.seasonality": "定义周期结构、原子数量以及小波相关设置。",
    "stage1.noise": "控制噪声强度和波动率突增窗口。",
    "causal": "控制 DAG 稀疏度和 ARX 动力学参数。",
    "anomaly": "控制异常事件数量和允许的异常原型。",
    "debug": "用于临时关闭某个阶段，便于单独调试。",
}

FIELD_DESCRIPTIONS = {
    "min": "Lower bound of the sampled range.",
    "max": "Upper bound of the sampled range.",
    "num_samples": "How many samples to generate in one run.",
    "sequence_length": "Allowed sequence length range for each sample.",
    "anomaly_sample_ratio": "Probability that a sampled series instance contains anomalies.",
    "num_series": "How many variables/nodes each sample contains.",
    "seed": "Random seed used for reproducible sampling.",
    "change_points": "How many slope changes a piecewise trend may contain.",
    "slope_scale": "Typical magnitude of trend slope changes.",
    "arima_noise_scale": "Innovation scale used in ARIMA-like trend generation.",
    "p_max": "Maximum AR order when sampling ARIMA-like trends.",
    "q_max": "Maximum MA order when sampling ARIMA-like trends.",
    "d": "Allowed differencing orders for ARIMA-like trends.",
    "coef_bound": "Absolute bound applied to sampled AR/MA coefficients.",
    "atoms": "How many seasonal atoms can be combined in one node.",
    "amplitude": "Allowed amplitude range for generated atoms.",
    "base_period": "Controls the period ranges used for seasonal atoms.",
    "low": "Range used when sampling lower-frequency periods.",
    "high": "Range used when sampling higher-frequency periods.",
    "families": "Sampling weights for each wavelet family.",
    "scale": "Allowed scale range for wavelet atoms.",
    "shift": "Allowed phase-shift or translation range for wavelet atoms.",
    "ratio": "Chance of creating a contrastive wavelet pair instead of a single atom.",
    "params": "Which wavelet parameters are allowed to change inside a contrastive pair.",
    "sigma": "Base standard deviation assigned to each noise regime.",
    "volatility_windows": "How many volatility-change windows may be inserted.",
    "volatility_multiplier": "Range for scaling volatility inside burst windows.",
    "num_nodes": "Valid node-count range for the causal graph.",
    "edge_density": "Expected sparsity of the sampled DAG.",
    "max_lag": "Maximum lag used on causal edges.",
    "a_i_bound": "Absolute bound of self-dynamics coefficients in ARX.",
    "bias_std": "Standard deviation of causal bias terms.",
    "b_ij_std": "Standard deviation of sampled edge gains.",
    "alpha_i_min": "Lower bound of the baseline-vs-latent mixing factor.",
    "alpha_i_max": "Upper bound of the baseline-vs-latent mixing factor.",
    "defaults": "Shared defaults for this configuration block.",
    "local": "Planner settings for additive local anomaly operators.",
    "seasonal": "Planner settings for seasonal-component anomaly operators.",
    "budget": "Controls how many events this anomaly family may contribute.",
    "allow_overlap": "Whether multiple events may occupy overlapping windows on the same node.",
    "min_gap": "Minimum empty gap kept between non-overlapping events on the same node.",
    "max_events_per_node": "Upper bound on how many events can be assigned to one node.",
    "events_per_sample": "How many events this anomaly family may sample in one series.",
    "activation_p": "Chance of enabling the seasonal anomaly planner for the current sample.",
    "window_length": "Allowed time-window length for anomaly events.",
    "type_weights": "Relative sampling weights across anomaly archetypes or operators.",
    "per_type": "Per-operator parameter ranges and enable flags.",
    "enabled": "Turns this specific anomaly operator on or off.",
    "endogenous_p": "Chance that sampled events from this family are injected before causal propagation.",
    "target_component": "Component or signal buffer that this anomaly family is intended to modify.",
    "node_policy": "Rule used to choose which nodes are eligible targets.",
    "mode": "Sampling mode used inside the node-selection policy.",
    "applies_to": "Seasonality families that are allowed to use this operator.",
    "half_width": "Half-width used by triangular spike templates.",
    "spike_count": "How many spikes appear inside one burst event.",
    "stride": "Spacing between neighboring spikes inside a burst.",
    "rise_length": "Length of the rising edge in a wide spike template.",
    "fall_length": "Length of the falling edge in a wide spike template.",
    "kappa": "Slope of the sigmoid used in sudden level shifts.",
    "rapid_tau": "Fast time constant used by asymmetric transient templates.",
    "slow_tau": "Slow time constant used by asymmetric transient templates.",
    "shift_magnitude": "Post-spike level-shift magnitude used by interaction templates.",
    "frequency": "Frequency range used by the shake operator.",
    "positive_p": "Chance that the plateau sign is sampled as positive.",
    "min_abs_amplitude": "Smallest absolute shock allowed when sampling signed amplitudes.",
    "factor": "Multiplicative change factor used by the operator.",
    "delta_phase": "Fixed phase offset applied inside the anomaly window.",
    "amplitude_scale": "Relative amplitude used when synthesizing a new harmonic.",
    "depth": "Target modulation depth used for harmonic modulation edits.",
    "delta_cycle": "Cycle-space shift applied to pulse-style seasonal waves.",
    "target_type_weights": "Sampling weights over target waveform types.",
    "target_family_weights": "Sampling weights over replacement wavelet families.",
    "family_weights": "Sampling weights over new wavelet families.",
    "order": "Harmonic order used when adding a new periodic component.",
    "noise_scale": "Noise strength injected into the seasonal component window.",
    "enable_trend": "Toggle the trend component on or off for debugging.",
    "enable_seasonality": "Toggle the seasonality component on or off for debugging.",
    "enable_noise": "Toggle the noise component on or off for debugging.",
    "enable_causal": "Toggle DAG + ARX causal mixing on or off.",
    "enable_local_anomaly": "Toggle local anomaly sampling and injection.",
    "enable_seasonal_anomaly": "Toggle seasonal anomaly sampling and injection.",
}

FIELD_DESCRIPTIONS_ZH = {
    "min": "该范围的下界。",
    "max": "该范围的上界。",
    "num_samples": "一次生成任务会产出多少个样本。",
    "sequence_length": "每个样本允许的序列长度范围。",
    "anomaly_sample_ratio": "一个样本被采成异常样本的概率。",
    "num_series": "每个样本包含多少条变量序列。",
    "seed": "用于复现实验的随机种子。",
    "change_points": "分段趋势里允许出现多少个斜率变化点。",
    "slope_scale": "趋势斜率变化的典型尺度。",
    "arima_noise_scale": "ARIMA 风格趋势的创新噪声尺度。",
    "p_max": "ARIMA 风格趋势中 AR 阶数的上限。",
    "q_max": "ARIMA 风格趋势中 MA 阶数的上限。",
    "d": "ARIMA 风格趋势允许的差分阶数范围。",
    "coef_bound": "AR/MA 系数采样时的绝对值上界。",
    "atoms": "一个节点里最多组合多少个季节原子。",
    "amplitude": "季节原子的振幅范围。",
    "base_period": "控制季节原子的基础周期区间。",
    "low": "采样低频周期时使用的范围。",
    "high": "采样高频周期时使用的范围。",
    "families": "各小波族被选中的权重。",
    "scale": "小波原子的尺度范围。",
    "shift": "小波原子的平移或相位偏移范围。",
    "ratio": "把小波原子采成对比配对的概率。",
    "params": "允许在对比配对中改变的小波参数。",
    "sigma": "各噪声等级对应的基础标准差。",
    "volatility_windows": "允许插入多少个波动率变化窗口。",
    "volatility_multiplier": "窗口内波动率放大的范围。",
    "num_nodes": "因果图允许的节点数范围。",
    "edge_density": "采样 DAG 时的边密度。",
    "max_lag": "因果边上允许的最大滞后。",
    "a_i_bound": "ARX 中自动态系数的绝对值上界。",
    "bias_std": "因果偏置项的标准差。",
    "b_ij_std": "边增益采样的标准差。",
    "alpha_i_min": "基线与潜变量混合系数的下界。",
    "alpha_i_max": "基线与潜变量混合系数的上界。",
    "defaults": "这一层配置块共享的默认策略。",
    "local": "用于规划局部加性异常的配置。",
    "seasonal": "用于规划季节组件异常的配置。",
    "budget": "控制该异常族在单条样本里最多采多少事件。",
    "allow_overlap": "同一节点上的事件窗口是否允许重叠。",
    "min_gap": "同一节点相邻事件之间至少保留多少空白间隔。",
    "max_events_per_node": "单个节点允许承载的事件上限。",
    "events_per_sample": "该异常族在单条序列中采样多少个事件。",
    "activation_p": "当前样本启用季节异常规划器的概率。",
    "window_length": "异常事件作用窗口的长度范围。",
    "type_weights": "不同异常类型被采到的相对权重。",
    "per_type": "每一种异常类型自己的开关和参数范围。",
    "enabled": "是否启用这一类异常。",
    "endogenous_p": "该异常族在因果传播前注入的概率。",
    "target_component": "这一类异常语义上作用到哪个组件或缓冲层。",
    "node_policy": "如何选择异常落在哪些节点上的规则。",
    "mode": "节点采样策略的模式。",
    "applies_to": "这个季节异常算子适用的季节波形类型。",
    "half_width": "三角尖峰模板的半宽范围。",
    "spike_count": "一次 burst 内会出现多少个尖峰。",
    "stride": "burst 内相邻尖峰之间的步长。",
    "rise_length": "宽尖峰上升段的长度范围。",
    "fall_length": "宽尖峰下降段的长度范围。",
    "kappa": "突增或突降里 sigmoid 的斜率范围。",
    "rapid_tau": "非对称瞬态中的快时间常数范围。",
    "slow_tau": "非对称瞬态中的慢时间常数范围。",
    "shift_magnitude": "spike 后 level shift 的幅值范围。",
    "frequency": "shake 高频扰动的频率范围。",
    "positive_p": "普通 plateau 取正号的概率。",
    "min_abs_amplitude": "有符号冲击允许的最小绝对幅值。",
    "factor": "算子使用的乘性变化因子。",
    "delta_phase": "窗口内施加的固定相位偏移范围。",
    "amplitude_scale": "添加谐波时相对原始幅值的比例范围。",
    "depth": "调制深度的目标范围。",
    "delta_cycle": "脉冲型波形的周期平移范围。",
    "target_type_weights": "目标波形类型的采样权重。",
    "target_family_weights": "替换用小波族的采样权重。",
    "family_weights": "新增小波原子时使用的小波族权重。",
    "order": "新增谐波时使用的阶数范围。",
    "noise_scale": "注入季节窗口噪声时的强度范围。",
    "enable_trend": "调试时是否启用趋势组件。",
    "enable_seasonality": "调试时是否启用季节组件。",
    "enable_noise": "调试时是否启用噪声组件。",
    "enable_causal": "是否启用 DAG + ARX 因果混合。",
    "enable_local_anomaly": "是否启用局部异常采样和注入。",
    "enable_seasonal_anomaly": "是否启用季节异常采样和注入。",
}

def _build_select_options() -> dict[str, list[str]]:
    return {
        "anomaly.local.defaults.target_component": list(LOCAL_TARGET_COMPONENTS),
        "anomaly.local.defaults.node_policy.mode": list(LOCAL_NODE_POLICY_MODES),
        "anomaly.seasonal.defaults.target_component": list(SEASONAL_TARGET_COMPONENTS),
        "anomaly.seasonal.defaults.node_policy.mode": list(SEASONAL_NODE_POLICY_MODES),
    }


def _build_multi_select_options(defaults: dict[str, Any]) -> dict[str, list[str]]:
    options = {
        "stage1.seasonality.wavelet.contrastive.params": list(
            defaults["stage1"]["seasonality"]["wavelet"]["contrastive"]["params"]
        )
    }
    anomaly = defaults.get("anomaly", {})
    for family in ["local", "seasonal"]:
        per_type = anomaly.get(family, {}).get("per_type", {})
        for kind, spec in per_type.items():
            applies_to = spec.get("applies_to")
            if isinstance(applies_to, list) and applies_to:
                options[f"anomaly.{family}.per_type.{kind}.applies_to"] = [str(value) for value in applies_to]
    return options

RANGE_BOUNDS: dict[str, tuple[float, float, str]] = {
    "sequence_length": (32, 2048, "int"),
    "num_series": (1, 12, "int"),
    "stage1.trend.change_points": (1, 8, "int"),
    "stage1.trend.arima.d": (1, 3, "int"),
    "stage1.seasonality.atoms": (1, 5, "int"),
    "stage1.seasonality.amplitude": (0.1, 3.0, "float"),
    "stage1.seasonality.base_period.low": (24, 256, "int"),
    "stage1.seasonality.base_period.high": (4, 64, "int"),
    "stage1.seasonality.wavelet.scale": (0.04, 0.5, "float"),
    "stage1.seasonality.wavelet.shift": (0.0, 1.0, "float"),
    "stage1.noise.volatility_windows": (0, 6, "int"),
    "stage1.noise.volatility_multiplier": (0.05, 1.5, "float"),
    "causal.num_nodes": (1, 20, "int"),
    "anomaly.local.budget.events_per_sample": (0, 5, "int"),
    "anomaly.local.defaults.window_length": (1, 160, "int"),
    "anomaly.seasonal.budget.events_per_sample": (0, 4, "int"),
    "anomaly.seasonal.defaults.window_length": (1, 160, "int"),
}

SCALAR_BOUNDS: dict[str, tuple[float, float, str]] = {
    "num_samples": (1, 256, "int"),
    "seed": (1, 999_999, "int"),
    "anomaly_sample_ratio": (0.0, 1.0, "float"),
    "stage1.trend.slope_scale": (0.001, 0.08, "float"),
    "stage1.trend.arima_noise_scale": (0.005, 0.2, "float"),
    "stage1.trend.arima.p_max": (0, 4, "int"),
    "stage1.trend.arima.q_max": (0, 4, "int"),
    "stage1.trend.arima.coef_bound": (0.1, 1.2, "float"),
    "stage1.seasonality.wavelet.contrastive.ratio": (0.0, 1.0, "float"),
    "causal.edge_density": (0.0, 0.8, "float"),
    "causal.max_lag": (0, 24, "int"),
    "causal.a_i_bound": (0.1, 1.2, "float"),
    "causal.bias_std": (0.0, 0.5, "float"),
    "causal.b_ij_std": (0.05, 0.8, "float"),
    "causal.alpha_i_min": (0.05, 0.8, "float"),
    "causal.alpha_i_max": (0.2, 0.98, "float"),
    "anomaly.defaults.min_gap": (0, 32, "int"),
    "anomaly.defaults.max_events_per_node": (1, 8, "int"),
    "anomaly.local.defaults.endogenous_p": (0.0, 1.0, "float"),
    "anomaly.seasonal.defaults.endogenous_p": (0.0, 1.0, "float"),
    "anomaly.seasonal.activation_p": (0.0, 1.0, "float"),
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _deep_copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _merge_import_payload(default_node: Any, imported_node: Any) -> Any:
    if isinstance(default_node, dict):
        if not isinstance(imported_node, dict):
            if set(default_node.keys()) == {"min", "max"} and isinstance(imported_node, (int, float)):
                return {"min": imported_node, "max": imported_node}
            return imported_node

        merged: dict[str, Any] = {}
        for key, default_value in default_node.items():
            if key in imported_node:
                merged[key] = _merge_import_payload(default_value, imported_node[key])
            else:
                merged[key] = _deep_copy_jsonable(default_value)
        for key, value in imported_node.items():
            if key not in merged:
                merged[key] = _to_jsonable(value)
        return merged

    return _to_jsonable(imported_node)


def _pretty_label(path: str, locale: str = "en") -> str:
    path_overrides = PATH_LABEL_OVERRIDES_ZH if locale == "zh" else PATH_LABEL_OVERRIDES
    section_labels = SECTION_LABELS_ZH if locale == "zh" else SECTION_LABELS
    field_labels = FIELD_LABELS_ZH if locale == "zh" else FIELD_LABELS
    if path in path_overrides:
        return path_overrides[path]
    if path in section_labels:
        return section_labels[path]
    leaf = path.split(".")[-1]
    if leaf in field_labels:
        return field_labels[leaf]
    return leaf.replace("_", " ").title()


def _describe_path(path: str, locale: str = "en") -> str:
    section_descriptions = SECTION_DESCRIPTIONS_ZH if locale == "zh" else SECTION_DESCRIPTIONS
    field_descriptions = FIELD_DESCRIPTIONS_ZH if locale == "zh" else FIELD_DESCRIPTIONS
    if path in section_descriptions:
        return section_descriptions[path]
    leaf = path.split(".")[-1]
    if path.startswith("weights.") and leaf not in {"weights", "seasonality_type", "trend_type", "frequency_regime", "noise_level"}:
        return "This weight controls how often the option is sampled." if locale == "en" else "这个权重决定对应选项被采样到的频率。"
    if leaf in field_descriptions:
        return field_descriptions[leaf]
    return "Configure this parameter." if locale == "en" else "设置该参数。"


def _build_locale_payload(defaults: dict[str, Any], locale: str) -> dict[str, Any]:
    path_labels = {path: _pretty_label(path, locale=locale) for path in _collect_paths(defaults)}
    path_descriptions = {path: _describe_path(path, locale=locale) for path in _collect_paths(defaults)}
    path_labels["root"] = SECTION_LABELS_ZH["root"] if locale == "zh" else SECTION_LABELS["root"]
    path_descriptions["root"] = _describe_path("root", locale=locale)
    return {
        "pathLabels": path_labels,
        "pathDescriptions": path_descriptions,
    }


def _build_numeric_bounds() -> dict[str, dict[str, int | float | str]]:
    bounds: dict[str, dict[str, int | float | str]] = {}
    for path, (low, high, kind) in SCALAR_BOUNDS.items():
        bounds[path] = {"min": low, "max": high, "kind": kind}
    for path, (low, high, kind) in RANGE_BOUNDS.items():
        bounds[f"{path}.min"] = {"min": low, "max": high, "kind": kind}
        bounds[f"{path}.max"] = {"min": low, "max": high, "kind": kind}
    return bounds


def _collect_paths(node: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if prefix:
        paths.append(prefix)
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_collect_paths(value, child_prefix))
    return paths


def load_default_config_raw() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    return _deep_copy_jsonable(DEFAULT_CONFIG)


def get_bootstrap_payload() -> dict[str, Any]:
    defaults = load_default_config_raw()
    multi_select_options = _build_multi_select_options(defaults)
    select_options = _build_select_options()
    return {
        "defaults": defaults,
        "ui": {
            "sectionLabels": SECTION_LABELS,
            "fieldLabels": FIELD_LABELS,
            "multiSelectOptions": multi_select_options,
            "selectOptions": select_options,
            "numericBounds": _build_numeric_bounds(),
            "pathLabels": _build_locale_payload(defaults, locale="en")["pathLabels"],
            "pathDescriptions": _build_locale_payload(defaults, locale="en")["pathDescriptions"],
            "locales": {
                "en": _build_locale_payload(defaults, locale="en"),
                "zh": _build_locale_payload(defaults, locale="zh"),
            },
        },
    }


def import_config_text(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        raise ValueError("Config text is empty.")

    parsed: Any
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ValueError("YAML import requires PyYAML to be installed.") from exc
        parsed = yaml.safe_load(payload)

    if not isinstance(parsed, dict):
        raise ValueError("Imported config must be a JSON/YAML object.")

    defaults = load_default_config_raw()
    merged = _merge_import_payload(defaults, parsed)
    cfg = load_config_from_raw(merged)
    return {"config": _to_jsonable(cfg.raw)}


def randomize_config(seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    for _ in range(128):
        candidate = _randomize_from_defaults(rng)
        try:
            load_config_from_raw(candidate)
            return candidate
        except ValueError:
            continue
    raise ValueError("Unable to create a valid randomized config after 128 attempts.")


def _randomize_from_defaults(rng: np.random.Generator) -> dict[str, Any]:
    raw = _deep_copy_jsonable(load_default_config_raw())
    multi_select_options = _build_multi_select_options(raw)
    raw["num_samples"] = _sample_scalar("num_samples", rng)
    raw["seed"] = _sample_scalar("seed", rng)
    raw["anomaly_sample_ratio"] = _sample_scalar("anomaly_sample_ratio", rng)
    raw["sequence_length"] = _sample_range("sequence_length", rng)
    raw["num_series"] = _sample_range("num_series", rng)

    raw["weights"] = {key: _randomize_weight_dict(weights, rng) for key, weights in raw["weights"].items()}

    trend = raw["stage1"]["trend"]
    trend["change_points"] = _sample_range("stage1.trend.change_points", rng)
    trend["slope_scale"] = _sample_scalar("stage1.trend.slope_scale", rng)
    trend["arima_noise_scale"] = _sample_scalar("stage1.trend.arima_noise_scale", rng)
    trend["arima"]["p_max"] = _sample_scalar("stage1.trend.arima.p_max", rng)
    trend["arima"]["q_max"] = _sample_scalar("stage1.trend.arima.q_max", rng)
    trend["arima"]["d"] = _sample_range("stage1.trend.arima.d", rng)
    trend["arima"]["coef_bound"] = _sample_scalar("stage1.trend.arima.coef_bound", rng)

    season = raw["stage1"]["seasonality"]
    season["atoms"] = _sample_range("stage1.seasonality.atoms", rng)
    season["amplitude"] = _sample_range("stage1.seasonality.amplitude", rng)
    season["base_period"]["high"] = _sample_range("stage1.seasonality.base_period.high", rng)
    low_period = _sample_range("stage1.seasonality.base_period.low", rng)
    high_max = int(season["base_period"]["high"]["max"])
    low_period["min"] = max(int(low_period["min"]), high_max + 1)
    low_period["max"] = max(int(low_period["max"]), int(low_period["min"]))
    season["base_period"]["low"] = low_period
    season["wavelet"]["families"] = _randomize_weight_dict(season["wavelet"]["families"], rng)
    season["wavelet"]["scale"] = _sample_range("stage1.seasonality.wavelet.scale", rng)
    season["wavelet"]["shift"] = _sample_range("stage1.seasonality.wavelet.shift", rng)
    season["wavelet"]["contrastive"]["ratio"] = _sample_scalar(
        "stage1.seasonality.wavelet.contrastive.ratio",
        rng,
    )
    season["wavelet"]["contrastive"]["params"] = _random_subset(
        multi_select_options["stage1.seasonality.wavelet.contrastive.params"],
        rng,
    )

    noise = raw["stage1"]["noise"]
    noise["sigma"] = _randomize_scalar_dict(noise["sigma"], 0.001, 0.4, rng)
    noise["volatility_windows"] = _sample_range("stage1.noise.volatility_windows", rng)
    noise["volatility_multiplier"] = _sample_range("stage1.noise.volatility_multiplier", rng)

    causal = raw["causal"]
    causal["num_nodes"] = _sample_range("causal.num_nodes", rng)
    num_series = raw["num_series"]
    causal["num_nodes"]["min"] = min(int(causal["num_nodes"]["min"]), int(num_series["min"]))
    causal["num_nodes"]["max"] = max(int(causal["num_nodes"]["max"]), int(num_series["max"]))
    causal["edge_density"] = _sample_scalar("causal.edge_density", rng)
    causal["max_lag"] = _sample_scalar("causal.max_lag", rng)
    causal["a_i_bound"] = _sample_scalar("causal.a_i_bound", rng)
    causal["bias_std"] = _sample_scalar("causal.bias_std", rng)
    causal["b_ij_std"] = _sample_scalar("causal.b_ij_std", rng)
    alpha_min = _sample_scalar("causal.alpha_i_min", rng)
    alpha_max = _sample_scalar("causal.alpha_i_max", rng)
    causal["alpha_i_min"] = min(alpha_min, alpha_max)
    causal["alpha_i_max"] = max(alpha_min, alpha_max)

    anomaly = raw["anomaly"]
    anomaly["defaults"]["allow_overlap"] = bool(rng.random() < 0.25)
    anomaly["defaults"]["min_gap"] = _sample_scalar("anomaly.defaults.min_gap", rng)
    anomaly["defaults"]["max_events_per_node"] = _sample_scalar("anomaly.defaults.max_events_per_node", rng)
    _randomize_anomaly_family(
        anomaly["local"],
        budget_path="anomaly.local.budget.events_per_sample",
        window_path="anomaly.local.defaults.window_length",
        endogenous_path="anomaly.local.defaults.endogenous_p",
        rng=rng,
    )
    anomaly["seasonal"]["activation_p"] = _sample_scalar("anomaly.seasonal.activation_p", rng)
    _randomize_anomaly_family(
        anomaly["seasonal"],
        budget_path="anomaly.seasonal.budget.events_per_sample",
        window_path="anomaly.seasonal.defaults.window_length",
        endogenous_path="anomaly.seasonal.defaults.endogenous_p",
        rng=rng,
    )

    debug = raw["debug"]
    for key in list(debug.keys()):
        debug[key] = bool(rng.random() < 0.82)
    if not any([debug["enable_trend"], debug["enable_seasonality"], debug["enable_noise"]]):
        debug["enable_trend"] = True
    if not debug["enable_local_anomaly"] and not debug["enable_seasonal_anomaly"]:
        debug["enable_local_anomaly"] = True

    return raw


def _sample_scalar(path: str, rng: np.random.Generator) -> int | float:
    low, high, kind = SCALAR_BOUNDS[path]
    if kind == "int":
        return int(rng.integers(int(low), int(high) + 1))
    return round(float(rng.uniform(low, high)), 4)


def _sample_range(path: str, rng: np.random.Generator) -> dict[str, int | float]:
    low, high, kind = RANGE_BOUNDS[path]
    if kind == "int":
        start = int(rng.integers(int(low), int(high) + 1))
        end = int(rng.integers(start, int(high) + 1))
        return {"min": start, "max": end}
    start = round(float(rng.uniform(low, high)), 4)
    end = round(float(rng.uniform(start, high)), 4)
    return {"min": start, "max": end}


def _randomize_weight_dict(weights: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    sampled = {key: float(rng.uniform(0.05, 1.0)) for key in weights}
    total = sum(sampled.values())
    return {key: round(value / total, 4) for key, value in sampled.items()}


def _randomize_scalar_dict(values: dict[str, float], low: float, high: float, rng: np.random.Generator) -> dict[str, float]:
    return {key: round(float(rng.uniform(low, high)), 4) for key in values}


def _random_subset(options: list[str], rng: np.random.Generator) -> list[str]:
    count = int(rng.integers(1, len(options) + 1))
    chosen = list(rng.choice(options, size=count, replace=False))
    return sorted(str(value) for value in chosen)


def _randomize_anomaly_family(
    family: dict[str, Any],
    *,
    budget_path: str,
    window_path: str,
    endogenous_path: str,
    rng: np.random.Generator,
) -> None:
    family["budget"]["events_per_sample"] = _sample_range(budget_path, rng)
    family["defaults"]["window_length"] = _sample_range(window_path, rng)
    family["defaults"]["endogenous_p"] = _sample_scalar(endogenous_path, rng)
    family["type_weights"] = _randomize_weight_dict(family["type_weights"], rng)

    enabled_any = False
    for spec in family.get("per_type", {}).values():
        spec["enabled"] = bool(rng.random() < 0.82)
        enabled_any = enabled_any or bool(spec["enabled"])
        for key, value in list(spec.items()):
            if key == "enabled":
                continue
            if isinstance(value, dict) and set(value.keys()) == {"min", "max"} and not isinstance(value.get("min"), bool):
                # Keep detailed numeric ranges stable; the high-level budget/default knobs already randomize aggressively.
                continue
            if isinstance(value, dict):
                spec[key] = _randomize_weight_dict(value, rng)
    if not enabled_any and family.get("per_type"):
        first_key = next(iter(family["per_type"]))
        family["per_type"][first_key]["enabled"] = True


def preview_sample(raw_config: dict[str, Any]) -> dict[str, Any]:
    preview_raw = _deep_copy_jsonable(raw_config)
    preview_raw["num_samples"] = 1
    cfg = load_config_from_raw(preview_raw)

    pipeline = SyntheticGeneratorPipeline(cfg)
    rng = np.random.default_rng(cfg.seed)
    n, d = pipeline._sample_dimensions(rng)
    t = np.arange(n, dtype=float)

    stage1_params = pipeline._sample_stage1_params(n=n, d=d, rng=rng)
    stage1_components = _realize_stage1_preview(cfg, t=t, stage1_params=stage1_params)
    x_stage1 = stage1_components["stage1_baseline"]

    if cfg.debug.enable_causal:
        graph = CausalGraphSampler(cfg).sample_graph(num_nodes=d, rng=rng)
        arx = ARXSystem(cfg, graph)
        arx_params = arx.sample_params(rng)
    else:
        graph = pipeline._empty_graph(d)
        arx = ARXSystem(cfg, graph)
        arx_params = {"disabled": True}

    local_injector = LocalAnomalyInjector(cfg)
    seasonal_injector = SeasonalAnomalyInjector(cfg)
    sampled_local_events: list[AnomalyEvent] = []
    sampled_seasonal_events: list[AnomalyEvent] = []

    if rng.random() < cfg.anomaly_sample_ratio:
        if cfg.debug.enable_local_anomaly:
            sampled_local_events = local_injector.sample_events(n=n, d=d, rng=rng, graph=graph)
        if cfg.debug.enable_seasonal_anomaly:
            sampled_seasonal_events = seasonal_injector.sample_events(
                n=n,
                d=d,
                rng=rng,
                stage1_params=stage1_params,
            )

    if not cfg.debug.enable_causal:
        for event in sampled_local_events:
            event.is_endogenous = False
            event.root_cause_node = None
        for event in sampled_seasonal_events:
            event.is_endogenous = False
            event.root_cause_node = None

    pre_causal_local_events = [event for event in sampled_local_events if bool(event.is_endogenous)]
    post_causal_local_events = [event for event in sampled_local_events if not bool(event.is_endogenous)]

    x_stage1_anom = x_stage1.copy()
    realized_events: list[AnomalyEvent] = []
    if pre_causal_local_events:
        x_stage1_anom, local_events = local_injector.apply_events(x_normal=x_stage1_anom, events=pre_causal_local_events)
        realized_events.extend(local_events)
    pre_causal_local_delta = x_stage1_anom - x_stage1

    if cfg.debug.enable_causal:
        x_stage2_normal, causal_state = arx.simulate_with_params(x_base=x_stage1, n_steps=n, params=arx_params)
        x_observed, _ = arx.simulate_with_params(x_base=x_stage1_anom, n_steps=n, params=arx_params)
    else:
        x_stage2_normal = x_stage1.copy()
        x_observed = x_stage1_anom.copy()
        causal_state = pipeline._disabled_causal_state(n=n, d=d)
    x_after_causal = x_observed.copy()

    x_before_post_local = x_observed.copy()
    if post_causal_local_events:
        x_observed, local_events = local_injector.apply_events(x_normal=x_observed, events=post_causal_local_events)
        realized_events.extend(local_events)
    post_causal_local_delta = x_observed - x_before_post_local

    x_before_seasonal = x_observed.copy()
    if sampled_seasonal_events:
        x_observed, seasonal_events = seasonal_injector.apply_events(
            x_input=x_observed,
            events=sampled_seasonal_events,
            rng=rng,
            t=t,
            stage1_params=stage1_params,
            arx=arx if cfg.debug.enable_causal else None,
            arx_params=arx_params if cfg.debug.enable_causal else None,
        )
        realized_events.extend(seasonal_events)
    seasonal_delta = x_observed - x_before_seasonal
    causal_effect = x_stage2_normal - x_stage1
    final_anomaly_delta = x_observed - x_stage2_normal

    labels: LabelPayload = LabelBuilder(cfg).build(
        x_normal=x_stage2_normal,
        x_anom=x_observed,
        events=realized_events,
        graph=graph,
        causal_state=causal_state,
    )

    metadata: GenerationMetadata = {
        "sample": {"seed_state": str(rng.bit_generator.state["state"]["state"])},
        "stage1": {"params": stage1_params},
        "stage2": {"params": arx_params},
        "stage3": {
            "sampled_events": {
                "local": [event.to_record() for event in sampled_local_events],
                "seasonal": [event.to_record() for event in sampled_seasonal_events],
            }
        },
    }
    series_catalog = [
        {"id": "observed", "label": "Final Observed", "group": "Final", "kind": "signal"},
        {"id": "final_anomaly_delta", "label": "Observed - Stage2 Normal", "group": "Final", "kind": "delta"},
        {"id": "stage2_normal", "label": "Stage2 Normal", "group": "Causal", "kind": "signal"},
        {"id": "stage2_causal_effect", "label": "Stage2 Normal - Stage1 Baseline", "group": "Causal", "kind": "delta"},
        {"id": "stage3_pre_causal_local_delta", "label": "Pre-Causal Local Delta", "group": "Local", "kind": "delta"},
        {"id": "stage3_post_causal_local_delta", "label": "Post-Causal Local Delta", "group": "Local", "kind": "delta"},
        {"id": "stage3_seasonal_delta", "label": "Seasonal Delta", "group": "Seasonal", "kind": "delta"},
        {"id": "stage1_baseline", "label": "Stage1 Baseline", "group": "Stage1", "kind": "signal"},
        {"id": "stage1_trend", "label": "Stage1 Trend", "group": "Stage1", "kind": "component"},
        {"id": "stage1_seasonality", "label": "Stage1 Seasonality", "group": "Stage1", "kind": "component"},
        {"id": "stage1_noise", "label": "Stage1 Noise", "group": "Stage1", "kind": "component"},
    ]
    series_payload = {
        **stage1_components,
        "stage2_normal": x_stage2_normal,
        "stage2_causal_effect": causal_effect,
        "stage3_pre_causal_local_delta": pre_causal_local_delta,
        "stage3_post_causal_local_delta": post_causal_local_delta,
        "stage3_seasonal_delta": seasonal_delta,
        "final_anomaly_delta": final_anomaly_delta,
        "observed": x_observed,
    }
    label_summary = labels["summary"]
    payload = {
        "summary": {
            "length": int(n),
            "num_series": int(d),
            "is_anomalous_sample": int(labels["is_anomalous_sample"]),
            "num_events": int(label_summary["total"]),
            "num_local_events": int(label_summary["local"]),
            "num_seasonal_events": int(label_summary["seasonal"]),
            "num_endogenous_events": int(label_summary["endogenous"]),
        },
        "series": series_payload,
        "series_catalog": series_catalog,
        "labels": {
            "point_mask": labels["point_mask"],
            "point_mask_any": labels["point_mask_any"],
            "root_cause": labels["root_cause"],
            "affected_nodes": labels["affected_nodes"],
            "events": labels["events"],
            "summary": label_summary,
        },
        "debug": {
            "series_stats": {key: _series_stats(value) for key, value in series_payload.items()},
            "stage_windows": {
                "pre_causal_local_nonzero": int(np.count_nonzero(np.abs(pre_causal_local_delta) > 1e-8)),
                "post_causal_local_nonzero": int(np.count_nonzero(np.abs(post_causal_local_delta) > 1e-8)),
                "seasonal_nonzero": int(np.count_nonzero(np.abs(seasonal_delta) > 1e-8)),
            },
        },
        "graph": {
            "num_nodes": int(graph.num_nodes),
            "adjacency": graph.adjacency,
            "topo_order": graph.topo_order,
            "parents": graph.parents,
        },
        "metadata": metadata,
    }
    return _to_jsonable(payload)


def _series_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "abs_max": float(np.max(np.abs(values))),
    }


def _realize_stage1_preview(cfg, t: np.ndarray, stage1_params: list[Stage1NodeParams]) -> dict[str, np.ndarray]:
    n = t.size
    d = len(stage1_params)
    trend_all = np.zeros((n, d), dtype=float)
    season_all = np.zeros((n, d), dtype=float)
    noise_all = np.zeros((n, d), dtype=float)
    x_base = np.zeros((n, d), dtype=float)
    for spec in stage1_params:
        node = int(spec["node"])
        trend = render_trend(t=t, params=spec["trend"]) if cfg.debug.enable_trend else np.zeros(n, dtype=float)
        season = (
            render_seasonality(t=t, params=spec["seasonality"])
            if cfg.debug.enable_seasonality
            else np.zeros(n, dtype=float)
        )
        noise = render_noise(n=n, params=spec["noise"]) if cfg.debug.enable_noise else np.zeros(n, dtype=float)
        trend_all[:, node] = trend
        season_all[:, node] = season
        noise_all[:, node] = noise
        x_base[:, node] = trend + season + noise
    return {
        "stage1_trend": trend_all,
        "stage1_seasonality": season_all,
        "stage1_noise": noise_all,
        "stage1_baseline": x_base,
    }

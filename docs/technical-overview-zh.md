# Support-Conditioned Vision 技术说明

## 项目定位

本项目面向工业图像的 Support 条件分类。推理时提供两类带标签参考图（A/B，业务中通常对应 NG/OK）和待检 Query。系统根据当前 Support 建立分类边界，再对 Query 输出 A/B 预测。

当前发布包包含通用 DINOv2 + Support QP 分类路线、桌面/网页界面、LoRA 训练入口和按产品目录选择任务的路由模块。圆柱数量分支包含推理代码，但缺少数量读出权重，且可靠性状态未通过，正式模式返回 REVIEW。

## 通用分类

- 输入为完整 RGB 图像，缩放至 448×448。
- 主干为冻结的 DINOv2 ViT-S/14 Register，隐藏维度 384，共 12 个 Transformer Block。
- Patch Size 为 14，得到 32×32 个 Patch。
- 输出 CLS 全局特征和按空间顺序排列的 1,024 个 Patch 特征。
- A、B 两类的总统计权重各占 0.5；CLS 与 Patch 分别计算中心和尺度。
- 归一化特征用于求解带自由截距的硬间隔线性分类器：

[
\min_{w,b}\frac{1}{2}\|w\|_2^2,\qquad y_i(w^Tz_i+b)\geq 1
]

当前实现使用 SciPy SLSQP 求解对偶问题，并根据 KKT 条件修正活动集。只有所有 Support 均满足 margin 约束，才接受新的注册结果。特征编码使用 FP32，QP 求解和分类使用 FP64。

## Support 覆盖

桌面端支持手动选择、随机抽取和注册档案恢复。默认随机注册数量为 A、B 各 5 张。

项目历史复测观察到：NG 和 OK 分别手动注册 5 张后，分类结果恢复到正常水平。主要原因是少量 Support 难以覆盖姿态、纹理、背景和异常形态。类别平衡归一化让两类总权重各占一半，因此增加 Support 的价值主要是扩大类内变化覆盖。

5 + 5 是项目观察到的操作建议，不是代码强制数量，也不能替代独立验证。

## LoRA 与训练

三组 LoRA 适配器作用于 DINOv2 第 10、11 个 Transformer Block 的 Attention Q/V 投影，rank=4、alpha=4，共 12,288 个可训练参数。DINOv2 主干保持冻结。

`train.py` 只训练 LoRA，不训练 DINOv2 主干，也不训练圆柱数量读出头。训练配置绑定独立数据清单、RGB 查找表和训练日程；示例配置默认不授权执行。

## 产品路由

路由实现位于 `src/task_selection/registration.py`。系统根据 `Difficult/<产品目录>` 或显式产品参数选择任务，不读取图像内容判断产品。

| 产品 | 路由 |
|---|---|
| 特斯拉、甲片、金属膨胀螺丝1 | GENERIC |
| 圆柱轴 | QUANTITY |
| 未知或冲突目录 | REVIEW |

GENERIC 使用 DINOv2 + Support QP。QUANTITY 指向圆柱数量研究分支。

## 圆柱数量分支现状

业务规则为 1 件 OK，0 件或 2 件 NG。当前读出类别域为 0、1、2，没有独立的 3+ 类别。

Parent400 LoRA 适配器 `support_optimized_2_training.pt` 已随包提供；独立数量读出 `quantity_parent400_lambda1.pt` 和其拟合脚本没有随包提供。当前数量注册只使用 OK Support 做正常数量自检，NG Support 图像内容没有进入数量读出。正式可靠性状态为 `QUANTITY_RELIABILITY_NOT_VERIFIED`，`deploy_allowed=false`；正式模式返回 REVIEW。

## 依赖与数据边界

官方 DINOv2 源码和主干权重由 `deployment/setup.py` 外置准备。仓库不包含真实工业图片、Support/Query 数据集和实验运行产物。数量读出权重缺失意味着圆柱数量研究分支不能从干净发布包独立复现。

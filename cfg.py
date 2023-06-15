small_target = 1e8
period = 365
pro_args = dict(q=0,
                period=period,
                T=period / 365,
                r=0.03,
                k_out_ratio=1.03,
                k_in_ratio=0.80,
                k_coupon=0.1,
                d_coupon=0.1,
                M=10000,
                small_target=small_target,  # 初始投资金额
                mgn_acc_ratio=0.2,  # 保证金账户存款
                cash_inv_rate=0.05,  # 现金账户利息收入
                mgn_inv_rate=0.015,  # 保证金账户利息收入
                fee_rate=0.003,  # 交易摩擦成本
                mgn_ratio_i=0.14,  # 初始保证金
                mgn_ratio_m=0.08,  # 维持保证金
                stop_loss_ratio=0.95,  # 止损线
                )
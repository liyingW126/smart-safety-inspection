"""
关灯科技实战任务 - 缺货分级运营分析
根问题：缺货高度集中在少数门店与品类，是库存管理资源分配不均，
        而非单一的"库存同步时滞"技术问题。
结论：按缺货率把门店/品类分级，优先把盘点、补货、同步资源投到高风险门店+品类。

口径：每次到位缺货损失 90 分钟。
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = "pick_tasks.csv"

df = pd.read_csv(CSV_PATH)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["stock_synced_at"] = pd.to_datetime(df["stock_synced_at"], errors="coerce")
df["lag_min"] = (df["created_at"] - df["stock_synced_at"]).dt.total_seconds() / 60
df["f"] = pd.to_numeric(df["is_fulfilled"], errors="coerce")
known = df[df["f"].isin([0, 1])].copy()
missing = known[known["f"] == 0]
n_miss = len(missing)

print("===== 数据概况 =====")
print(f"总任务 {len(df)} | 有结果 {len(known)} | 缺货 {n_miss} | 缺货率 {n_miss/len(known):.1%}")

# ---------- 1. 门店维度 ----------
print("\n===== 门店缺货率（降序）=====")
store = known.groupby("store_name").agg(
    总任务=("f", "size"),
    缺货=("f", lambda x: (x == 0).sum()),
    平均时滞分钟=("lag_min", "mean"),
).reset_index()
store["缺货率"] = store["缺货"] / store["总任务"]
store = store.sort_values("缺货率", ascending=False).reset_index(drop=True)
print(store.to_string(index=False))

# 门店缺货集中度（帕累托）
store["缺货累计占比"] = store["缺货"].cumsum() / store["缺货"].sum()
store["门店累计占比"] = np.arange(1, len(store) + 1) / len(store)
print("\n===== 门店缺货集中度 =====")
for i, row in store.iterrows():
    print(f"Top {i+1} 门店（{row['store_name']}）累计覆盖缺货 {row['缺货累计占比']:.0%}")

# ---------- 2. 品类维度 ----------
print("\n===== 品类缺货率（降序）=====")
cat = known.groupby("category").agg(
    总任务=("f", "size"),
    缺货=("f", lambda x: (x == 0).sum()),
).reset_index()
cat["缺货率"] = cat["缺货"] / cat["总任务"]
cat = cat.sort_values("缺货率", ascending=False).reset_index(drop=True)
print(cat.to_string(index=False))

cat["缺货累计占比"] = cat["缺货"].cumsum() / cat["缺货"].sum()
print("\n===== 品类缺货集中度 =====")
for i, row in cat.iterrows():
    print(f"Top {i+1} 品类（{row['category']}）累计覆盖缺货 {row['缺货累计占比']:.0%}")

# ---------- 3. 门店 × 品类 交叉热点 ----------
print("\n===== 门店 × 品类 缺货率热点（缺货率 Top10 组合）=====")
pivot = known.pivot_table(index="store_name", columns="category",
                          values="f", aggfunc=lambda x: (x == 0).mean())
stacked = pivot.stack().reset_index()
stacked.columns = ["门店", "品类", "缺货率"]
hot = stacked.sort_values("缺货率", ascending=False).head(10)
print(hot.to_string(index=False))

# ---------- 4. 诊断：高缺货门店 vs 低缺货门店 的时滞对比 ----------
print("\n===== 成因诊断：门店缺货率 与 平均时滞 的关系 =====")
corr = store["缺货率"].corr(store["平均时滞分钟"])
print(f"门店缺货率 vs 门店平均时滞 相关系数 = {corr:.2f}")
print(store[["store_name", "缺货率", "平均时滞分钟"]].sort_values("缺货率", ascending=False).to_string(index=False))

# ---------- 5. 分级建议 + 收益量化 ----------
print("\n===== 门店分级建议 =====")
def tier(rate):
    if rate >= 0.15: return "高风险(≥15%)"
    if rate >= 0.10: return "中风险(10-15%)"
    return "低风险(<10%)"
store["分级"] = store["缺货率"].apply(tier)
for t, g in store.groupby("分级"):
    print(f"{t}: {len(g)} 家门店，缺货 {g['缺货'].sum()} 个，占全部缺货 {g['缺货'].sum()/n_miss:.0%}")

# 收益：优先治理高风险门店能覆盖的空跑损失
high = store[store["分级"] == "高风险(≥15%)"]
print(f"\n若优先治理高风险门店：可覆盖缺货 {high['缺货'].sum()} 个 = "
      f"{high['缺货'].sum()*90} 分钟空跑损失（占全部 {n_miss*90} 分钟的 {high['缺货'].sum()/n_miss:.0%}）")

# ---------- 6. 出图 ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 门店缺货率
axes[0, 0].bar(store["store_name"], store["缺货率"], color="coral")
axes[0, 0].axhline(0.15, color="red", linestyle="--", label="高风险线 15%")
axes[0, 0].set_title("各门店缺货率")
axes[0, 0].set_ylabel("缺货率")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].legend()

# 品类缺货率
axes[0, 1].barh(cat["category"][::-1], cat["缺货率"][::-1], color="steelblue")
axes[0, 1].set_title("各品类缺货率")

# 门店缺货帕累托
axes[1, 0].plot(store["门店累计占比"], store["缺货累计占比"], marker="o")
axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1, 0].set_title("门店缺货集中度（帕累托）")
axes[1, 0].set_xlabel("门店累计占比")
axes[1, 0].set_ylabel("缺货累计占比")

# 门店缺货率 vs 平均时滞（散点）
axes[1, 1].scatter(store["平均时滞分钟"], store["缺货率"], s=100)
for _, r in store.iterrows():
    axes[1, 1].annotate(r["store_name"], (r["平均时滞分钟"], r["缺货率"]))
axes[1, 1].set_title("门店缺货率 vs 平均时滞（成因诊断）")
axes[1, 1].set_xlabel("平均时滞(分钟)")
axes[1, 1].set_ylabel("缺货率")

plt.tight_layout()
plt.savefig("charts_overview.png", dpi=120)
print("\n图表已保存: charts_overview.png")
print("完成。")

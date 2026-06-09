#!/usr/bin/env python3
import json, logging, os, sys
from datetime import datetime
from fetcher import fetch_all
from reporter import generate_html
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("daily")
def _load_prev_mpc():
    try:
        with open("output/result.json") as f: return json.load(f).get("market_panic",{}).get("mpc")
    except: return None
def run():
    logger.info("=== 美股短线分析日报 ===")
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Step 1/4: 拉取数据...")
    df, hist, vix, fng, failed, news_dict = fetch_all()
    if df.empty: print("❌ 今日数据不可用", file=sys.stderr); sys.exit(1)
    if failed: logger.warning(f"部分缺失: {failed}")
    logger.info(f"拉取完成: {len(df)}标的, VIX={vix}, F&G={fng}")
    logger.info("Step 2/4: 计算指标...")
    from analyzer import run_analysis
    prev = _load_prev_mpc()
    result = run_analysis(df, hist, vix, fng, prev, news_dict)
    # merge change_pct from fetcher into stocks
    for s in result.get("stocks",[]):
        row = df[df["ticker"]==s["ticker"]]
        if len(row): s["change_pct"] = float(row["change_pct"].iloc[0])
    logger.info("Step 3/4: 生成报告...")
    print(result.get("summary_markdown",""))
    os.makedirs("output", exist_ok=True)
    html_path = generate_html(result)
    logger.info(f"Dashboard: {html_path}")
    with open("output/result.json","w") as f: json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Step 4/4: 完成 ✅")
    return result
if __name__ == "__main__": run()

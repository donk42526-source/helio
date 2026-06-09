#!/usr/bin/env python3
import json, logging, os, sys
import pandas as pd
from datetime import datetime
from fetcher import fetch_all
from reporter import generate_html

class RoundFloatEncoder(json.JSONEncoder):
    def encode(self, o):
        return super().encode(self._round(o))
    def _round(self, obj):
        if isinstance(obj, float): return round(obj, 2)
        if isinstance(obj, dict): return {k: self._round(v) for k,v in obj.items()}
        if isinstance(obj, list): return [self._round(i) for i in obj]
        return obj
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
    # merge change_pct from fetcher into stocks, sort QQQ/SPY first
    for s in result.get("stocks",[]):
        row = df[df["ticker"]==s["ticker"]]
        if len(row):
            s["change_pct"] = float(row["change_pct"].iloc[0])
            mc = row["market_cap"].iloc[0] if "market_cap" in row.columns else None
            if mc and not pd.isna(mc):
                if mc >= 1e12: s["market_cap_fmt"] = f"{mc/1e12:.1f}万亿"
                elif mc >= 1e8: s["market_cap_fmt"] = f"{mc/1e8:.0f}亿"
                else: s["market_cap_fmt"] = f"${mc:,.0f}"
    result["stocks"] = sorted(result.get("stocks",[]),
        key=lambda s: (0 if s["ticker"]=="QQQ" else 1 if s["ticker"]=="SPY" else 2, -s.get("sts",0)))
    logger.info("Step 3/4: 生成报告...")
    print(result.get("summary_markdown",""))
    os.makedirs("output", exist_ok=True)
    html_path = generate_html(result)
    logger.info(f"Dashboard: {html_path}")
    with open("output/result.json","w") as f: json.dump(result, f, ensure_ascii=False, indent=2, cls=RoundFloatEncoder)
    logger.info("Step 4/4: 完成 ✅")
    return result
if __name__ == "__main__": run()

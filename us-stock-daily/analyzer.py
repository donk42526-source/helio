"""analyzer.py — 美股短线投机分析引擎 | Phase 2 · Data Analyst · 2026-06-09"""
import numpy as np, pandas as pd
from typing import Optional, Tuple
from datetime import date
try: from config import TICKER_NAMES
except: TICKER_NAMES = {}

def vix_to_score(vix): return 0 if vix<13 else 15 if vix<17 else 30 if vix<20 else 50 if vix<25 else 70 if vix<30 else 85 if vix<35 else 100

def compute_mpc(vix, fng, prev_mpc=None):
    if fng is None: fng = max(0, 100 - vix*3)  # fallback: VIX越高F&G越低
    vs, fs = vix_to_score(vix), 100-fng; mpc = round(0.55*vs+0.45*fs,1)
    level = "calm" if mpc<25 else "moderate" if mpc<45 else "elevated" if mpc<65 else "high" if mpc<80 else "extreme"
    chg = round(mpc-prev_mpc,1) if prev_mpc is not None else None
    return {"mpc":mpc,"level":level,"mpc_change":chg,
            "components":{"vix":{"value":vix,"score":vs},"fear_greed":{"value":fng,"score":fs}}}

def compute_rsi(closes, period=14):
    if len(closes)<period+1: return np.nan
    d=np.diff(closes[-(period+1):]); g=np.where(d>0,d,0); l=np.where(d<0,-d,0)
    ag,al=np.mean(g),np.mean(l); return 100.0 if al==0 else 100-(100/(1+ag/al))

def compute_macd(closes):
    if len(closes)<26: return np.nan,np.nan,np.nan,np.nan
    e12=pd.Series(closes).ewm(span=12,adjust=False).mean()
    e26=pd.Series(closes).ewm(span=26,adjust=False).mean()
    m=e12-e26; s=m.ewm(span=9,adjust=False).mean()
    return m.iloc[-1],s.iloc[-1],m.iloc[-1]-s.iloc[-1],(m.iloc[-2]-s.iloc[-2] if len(m)>=2 else np.nan)

def compute_bollinger(closes, period=20, mult=2.0):
    if len(closes)<period: return np.nan,np.nan,np.nan,np.nan,np.nan
    w=closes[-period:]; mid=np.mean(w); std=np.std(w,ddof=1); up,lo=mid+mult*std,mid-mult*std
    bw=(up-lo)/mid if mid else np.nan
    if len(closes)>=period+1:
        pw=closes[-(period+1):-1]; pmid=np.mean(pw); pstd=np.std(pw,ddof=1)
        pbw=((pmid+mult*pstd)-(pmid-mult*pstd))/pmid if pmid else np.nan
    else: pbw=np.nan
    return mid,up,lo,bw,pbw

def score_ma(c,ma20,ma50):
    if pd.isna(ma20) or pd.isna(ma50): return 50,"数据不足"
    if c>ma20>ma50: return 100,"Price>MA20>MA50"
    elif c>ma20: return 70,"Price>MA20但趋势未确认"
    elif ma20>c>ma50: return 40,"Price介于MA20/MA50间"
    elif c<ma20<ma50: return 10,"空头排列"
    return 0,"强下降趋势"

def score_rsi(rsi):
    if pd.isna(rsi): return 50,None
    if rsi>=75: return 25,rsi
    elif rsi>=65: return 75,rsi
    elif rsi>=55: return 65,rsi
    elif rsi>=45: return 50,rsi
    elif rsi>=40: return 55,rsi
    elif rsi>=35: return 60,rsi
    elif rsi>=30: return 50,rsi
    return 35,rsi

def score_macd(ml,sig,hist,prev_hist):
    if pd.isna(ml) or pd.isna(sig): return 50,"数据不足"
    ex=hist>prev_hist if not pd.isna(prev_hist) else None
    ct=hist<prev_hist if not pd.isna(prev_hist) else None
    if ml>sig:
        if ml>0:
            if ex is True: return 100,"金叉+柱扩大"
            elif ct is True: return 60,"金叉但柱缩小"
            return 80,"金叉"
        return 70,"底部金叉"
    else:
        if ml>0: return 35,"高位回调"
        if ct is True: return 25,"下跌减缓"
        return 0,"死叉"

def score_volume(vol,avg_vol,chg):
    if pd.isna(vol) or pd.isna(avg_vol) or avg_vol==0: return 50,"数据不足"
    r=vol/avg_vol
    if r>1.5 and chg>1: return 100,f"放量{round(r,1)}x上涨"
    elif r>1.5 and chg>=0: return 70,f"温和放量{round(r,1)}x"
    elif r>1.5: return 15,f"放量{round(r,1)}x下跌"
    elif r<0.7: return 35,"缩量"
    return 50,"正常交投"

def score_bollinger(c,u,l,m,bw,pbw):
    if pd.isna(c) or pd.isna(u) or pd.isna(l): return 50,"数据不足"
    nu, nl = c>=u*.98, c<=l*1.02
    ex=bw>pbw*1.1 if (not pd.isna(pbw) and pbw>0) else False
    ct=bw<pbw*.9 if (not pd.isna(pbw) and pbw>0) else False
    if ex and nu: return 80,"带宽扩张+近上轨"
    elif ex and nl: return 20,"带宽扩张+近下轨"
    elif ex: return 60,"带宽扩张方向待定"
    elif ct: return 50,"带宽收缩盘整"
    elif nl: return 70,"触下轨反弹"
    elif nu: return 30,"触上轨回落"
    return 50,"中性"

WEIGHTS={"ma":0.30,"rsi":0.25,"macd":0.20,"volume":0.15,"bollinger":0.10}

def compute_sts_one(t,c,ma20,ma50,vol,avol,chg,hist,target_price=None,recommendation=None):
    rv=compute_rsi(hist); ml,ms,mh,mph=compute_macd(hist); bm,bu,bl,bw,bpw=compute_bollinger(hist)
    sc={}; sc["ma"],md=score_ma(c,ma20,ma50); sc["rsi"],rv2=score_rsi(rv)
    sc["macd"],mcd=score_macd(ml,ms,mh,mph); sc["volume"],vd=score_volume(vol,avol,chg)
    sc["bollinger"],bd=score_bollinger(c,bu,bl,bm,bw,bpw)
    vw=sum(w for k,w in WEIGHTS.items() if not pd.isna(sc[k]))
    sts=sum(sc[k]*WEIGHTS[k]/vw for k in WEIGHTS if not pd.isna(sc[k])) if vw else 50; sts=round(sts,1)
    lv="strong" if sts>=70 else "bullish" if sts>=55 else "neutral" if sts>=40 else "weak" if sts>=25 else "bearish"
    signals_out={"ma":{"score":sc["ma"],"detail":md,"ma20":round(ma20,2) if not pd.isna(ma20) else None,"ma50":round(ma50,2) if not pd.isna(ma50) else None},
                 "rsi":{"score":sc["rsi"],"value":round(rv2,1) if rv2 else None,"raw":round(rv,1) if not pd.isna(rv) else None},
                 "macd":{"score":sc["macd"],"detail":mcd,"line":round(ml,4) if not pd.isna(ml) else None,"signal":round(ms,4) if not pd.isna(ms) else None,"histogram":round(mh,4) if not pd.isna(mh) else None},
                 "volume":{"score":sc["volume"],"detail":vd,"raw":int(vol),"avg_20d":int(avol)},
                 "bollinger":{"score":sc["bollinger"],"detail":bd,"upper":round(bu,2) if not pd.isna(bu) else None,"lower":round(bl,2) if not pd.isna(bl) else None,"mid":round(bm,2) if not pd.isna(bm) else None,"bandwidth":round(bw,4) if not pd.isna(bw) else None}}
    bias=compute_bias(signals_out)
    rs={"rs_5d":None,"rs_20d":None,"vs_qqq":None}
    analyst=score_target_price(c,target_price,recommendation)
    return {"ticker":t,"sts":sts,"level":lv,"signals":signals_out,"bias":bias,
            "relative_strength":rs,"analyst":analyst,"alerts":[],
            "close":c,"ma20_v":ma20 if not pd.isna(ma20) else None,"ma50_v":ma50 if not pd.isna(ma50) else None,
            "bb_upper":bu if not pd.isna(bu) else None,"bb_lower":bl if not pd.isna(bl) else None}

def compute_bias(signals):
    votes={"bull":0,"bear":0}
    if signals["ma"]["score"]>=70: votes["bull"]+=1
    elif signals["ma"]["score"]<=40: votes["bear"]+=1
    rv=signals["rsi"].get("value")
    if rv is not None:
        if 55<=rv<=75: votes["bull"]+=1
        elif rv<40: votes["bear"]+=1
    if signals["macd"]["score"]>=70: votes["bull"]+=1
    elif signals["macd"]["score"]<=30: votes["bear"]+=1
    if signals["volume"]["score"]>=70: votes["bull"]+=1
    elif signals["volume"]["score"]<=30: votes["bear"]+=1
    if signals["bollinger"]["score"]>=70: votes["bull"]+=1
    elif signals["bollinger"]["score"]<=30: votes["bear"]+=1
    direction="bullish" if votes["bull"]>=3 else "bearish" if votes["bear"]>=3 else "neutral"
    tv=votes["bull"]+votes["bear"]
    confidence="high" if tv>=4 and abs(votes["bull"]-votes["bear"])>=3 else "medium" if tv>=3 else "low"
    return {"direction":direction,"confidence":confidence,"votes":votes}

def score_target_price(close,tp,rec=None):
    if tp is None or pd.isna(tp) or tp<=0:
        return {"target_price":None,"upside_pct":None,"analyst_score":50,"analyst_rating":"N/A"}
    up=round((tp/close-1)*100,1)
    sc=85 if up>20 else 70 if up>10 else 60 if up>5 else 50 if up>0 else 40 if up>-5 else 30 if up>-10 else 15
    rm={"strong_buy":"强力买入","buy":"买入","hold":"持有","sell":"卖出","strong_sell":"强力卖出"}
    return {"target_price":round(tp,2),"upside_pct":up,"analyst_score":sc,"analyst_rating":rm.get(rec,"N/A")}

def generate_ticker_analysis(stock):
    t,sts=stock["ticker"],stock["sts"]; b=stock.get("bias",{}); rs=stock.get("relative_strength",{})
    a=stock.get("analyst",{}); d,c=b.get("direction","neutral"),b.get("confidence","medium")
    parts=[]
    if d=="bullish": parts.append(f"{'🔥' if c=='high' else '📈'} 短线偏多"+("(高置信)" if c=="high" else ""))
    elif d=="bearish": parts.append(f"{'⚠️' if c=='high' else '📉'} 短线偏空"+("(高置信)" if c=="high" else ""))
    else: parts.append("⚪ 方向不明")
    sd=[]; md=stock["signals"]["ma"]["detail"]; mcd=stock["signals"]["macd"]["detail"]
    if "MA20>MA50" in md: sd.append("多头排列")
    elif "空头" in md: sd.append("空头排列")
    if "金叉" in mcd: sd.append("MACD金叉")
    elif "死叉" in mcd: sd.append("MACD死叉")
    if sd: parts.append("，"+",".join(sd))
    vs=rs.get("vs_qqq"); rs5=rs.get("rs_5d")
    if vs and rs5 is not None:
        if vs=="outperform": parts.append(f"，跑赢QQQ {rs5:+.1f}%")
        elif vs=="underperform": parts.append(f"，跑输QQQ {rs5:+.1f}%")
    up=a.get("upside_pct")
    if up is not None: parts.append(f"，目标价上行{up:+.1f}%")
    nt = stock.get("news_text","")
    if nt: parts.append("\n"+nt)
    return f"**{t}** (STS {sts}): "+"".join(parts)

def generate_prediction(stock):
    """基于现有信号生成短线预测文字"""
    t=stock["ticker"]; sigs=stock["signals"]; bias=stock.get("bias",{})
    analyst=stock.get("analyst",{}); close=stock.get("close",0)
    lines=[f"🔮 **{t} 短线预测：**"]
    d=bias.get("direction","neutral"); c=bias.get("confidence","medium")
    rsi_v=sigs["rsi"].get("value"); macd_d=sigs["macd"]["detail"]
    if d=="bullish" and c=="high": lines.append("  • 方向：短期偏多，多头排列+MACD金叉确认，未来1-3日大概率延续强势")
    elif d=="bullish": lines.append("  • 方向：倾向看多但信号有分歧，若能放量突破前高则确认，否则可能震荡")
    elif d=="bearish" and c=="high": lines.append("  • 方向：短期偏空，MACD死叉，未来1-3日可能继续承压，不宜追低")
    elif d=="bearish": lines.append("  • 方向：偏弱但非极端，若RSI触及超卖或有短线反弹，但趋势未转前不宜重仓")
    else: lines.append("  • 方向：短期方向不明，多空力量均衡，可能在当前价位附近震荡整理")
    ma20=stock.get("ma20_v",0); ma50=stock.get("ma50_v",0)
    bu=stock.get("bb_upper",0); bl=stock.get("bb_lower",0)
    if ma20 and ma50:
        if close>ma20: sup,res=f"${ma20:.2f}(MA20)",f"${bu:.2f}(布林上轨)" if bu else "前高"
        elif close>ma50: sup,res=f"${ma50:.2f}(MA50)",f"${ma20:.2f}(MA20)"
        else: sup,res=f"${bl:.2f}(布林下轨)" if bl else "前低",f"${ma50:.2f}(MA50)"
        lines.append(f"  • 关键位：支撑 {sup}，阻力 {res}")
    if d=="bearish": lines.append("  • 反转条件：RSI回升至45以上+MACD出现金叉，则偏空转偏多")
    elif d=="bullish": lines.append("  • 反转条件：MACD出现死叉或跌破MA20，则偏多转偏空")
    else: lines.append("  • 突破信号：放量突破布林上轨则看多，跌破MA50则看空")
    up=analyst.get("upside_pct")
    if up is not None:
        if up>20 and d=="bearish": lines.append(f"  • ⚠️ 注意：分析师目标价上行{up:+.1f}%，技术面弱但基本面强——大跌可能是中长线买入机会")
        elif up>20 and d=="bullish": lines.append(f"  • ✅ 技术面+基本面共振：分析师目标价上行{up:+.1f}%，双重确认看多")
    if rsi_v is not None:
        if rsi_v>70: lines.append(f"  • ⚠️ RSI超买({rsi_v:.1f})，短期可能技术性回调，追高需谨慎")
        elif rsi_v<30: lines.append(f"  • ⚡ RSI超卖({rsi_v:.1f})，恐慌抛售往往不可持续，关注企稳反弹信号")
    return "\n".join(lines)
    """通俗白话解释，给不懂金融的人看"""
    t=stock["ticker"]; sigs=stock["signals"]; bias=stock.get("bias",{})
    rs=stock.get("relative_strength",{}); analyst=stock.get("analyst",{})
    lines=[f"💬 **{t} 通俗解读：**"]
    rsi_v=sigs["rsi"].get("value")
    if rsi_v is not None:
        if rsi_v>70: lines.append(f"  • RSI {rsi_v} — 处于超买区，最近涨太猛了，像抢购潮，短期可能要回调")
        elif rsi_v>60: lines.append(f"  • RSI {rsi_v} — 偏强势，买盘活跃但还没到过热")
        elif rsi_v>40: lines.append(f"  • RSI {rsi_v} — 中性区间，买卖力量均衡")
        elif rsi_v>30: lines.append(f"  • RSI {rsi_v} — 偏弱势，接近超卖区，像打折甩卖中")
        else: lines.append(f"  • RSI {rsi_v} — 处于超卖区，恐慌抛售中，但往往是短线反弹的前兆")
    macd_d=sigs["macd"]["detail"]
    if "金叉" in macd_d:
        if "柱扩大" in macd_d: lines.append(f"  • MACD 金叉且动能扩大 — 上涨加速中，短线势头正猛")
        else: lines.append(f"  • MACD 金叉 — 上涨趋势刚开始，像汽车刚挂上档起步")
    elif "死叉" in macd_d: lines.append(f"  • MACD 死叉 — 短期动能在减弱，像汽车在减速，注意刹车")
    ma_d=sigs["ma"]["detail"]
    if "Price>MA20>MA50" in ma_d: lines.append(f"  • 均线多头排列 — 短期在长期上方，趋势向好，像爬楼梯一阶比一阶高")
    elif "空头" in ma_d: lines.append(f"  • 均线空头排列 — 价格低于均线，趋势偏弱，像下楼梯")
    vol_d=sigs["volume"]["detail"]
    if "放量" in vol_d and "上涨" in vol_d: lines.append(f"  • 成交量放大且上涨 — 真金白银在买，不是虚涨，可信度高")
    elif "放量" in vol_d and "下跌" in vol_d: lines.append(f"  • 放量下跌 — 有大资金在出货，不是普通调整，要小心")
    bb_d=sigs["bollinger"]["detail"]
    if "扩张+近上轨" in bb_d: lines.append(f"  • 布林带扩张+近上轨 — 突破行情，波动加大，可能进入主升浪")
    elif "扩张+近下轨" in bb_d: lines.append(f"  • 布林带扩张+近下轨 — 破位风险，波动加大但方向向下")
    elif "收缩" in bb_d: lines.append(f"  • 布林带收缩 — 暴风雨前的宁静，盘整蓄力中，即将选择方向")
    vs=rs.get("vs_qqq"); rs5=rs.get("rs_5d")
    if vs=="outperform" and rs5 is not None: lines.append(f"  • 近5日跑赢纳斯达克 {rs5:+.1f}% — 比大盘更强，资金在流向这只票")
    elif vs=="underperform" and rs5 is not None: lines.append(f"  • 近5日跑输纳斯达克 {rs5:+.1f}% — 比大盘更弱，资金在流出")
    up=analyst.get("upside_pct"); rating=analyst.get("analyst_rating","N/A")
    if up is not None:
        if up>20: lines.append(f"  • 分析师目标价上行{up:+.1f}%({rating}) — 专业人士认为严重低估")
        elif up>5: lines.append(f"  • 分析师目标价上行{up:+.1f}%({rating}) — 专业人士认为有上涨空间")
        elif up>0: lines.append(f"  • 分析师目标价上行{up:+.1f}%({rating}) — 接近合理估值")
        else: lines.append(f"  • 分析师目标价下行{up:+.1f}%({rating}) — 专业人士认为当前偏贵")
    lines.append("")
    d=bias.get("direction","neutral"); c=bias.get("confidence","medium")
    if d=="bullish" and c=="high": lines.append(f"  ✅ 综合判断：多个信号一致看多，短线机会明确。")
    elif d=="bullish": lines.append(f"  📈 综合判断：偏多但信号有分歧，可小仓位试，设好止损。")
    elif d=="bearish" and c=="high": lines.append(f"  ⚠️ 综合判断：多个信号一致看空，短线不宜参与，等信号反转。")
    elif d=="bearish": lines.append(f"  📉 综合判断：偏空但非极端，已持有可考虑减仓，未持有先观望。")
    else: lines.append(f"  ⚪ 综合判断：方向不明朗，多看少动，等信号清晰了再出手。")
    if mpc and mpc.get("level") in ("high","extreme"):
        lines.append(f"  🔴 注意：当前市场恐慌度较高，即使看多的标的也要控制仓位、设紧止损。")
    return "\n".join(lines)

def detect_signals(mpc,stocks,df_daily):
    sigs=[]; nonb=[s for s in stocks if s["ticker"] not in ("SPY","QQQ")]; mc=mpc.get("mpc_change")
    if mc and mc>15:
        dips=[s["ticker"] for s in nonb if s["signals"]["rsi"].get("value") is not None and s["signals"]["rsi"]["value"]<35 and "放量" in s["signals"]["volume"]["detail"]]
        if len(dips)>=2: sigs.append({"type":"panic_dip","stocks":dips,"confidence":"high"})
    bo=[]
    for s in nonb:
        if s["sts"]>=75 and "放量" in s["signals"]["volume"]["detail"] and "金叉" in s["signals"]["macd"]["detail"]:
            s["alerts"].append("momentum_breakout"); bo.append(s["ticker"])
    if bo: sigs.append({"type":"momentum_breakout","stocks":bo,"count":len(bo)})
    bear=sum(1 for s in nonb if s["signals"]["macd"]["score"]<40)
    if bear>=5 and mpc["mpc"]>50:
        qr=df_daily[df_daily["ticker"]=="QQQ"]; qc=float(qr["change_pct"].iloc[0]) if len(qr) else None
        sigs.append({"type":"sector_alert","count":bear,"qqq_change":qc})
    qs=next((s["sts"] for s in stocks if s["ticker"]=="QQQ"),None)
    ss=next((s["sts"] for s in stocks if s["ticker"]=="SPY"),None)
    if qs is not None and ss is not None and qs<40 and ss>55:
        sn=sum(1 for s in nonb if s["sts"]>60)
        if sn>=3: sigs.append({"type":"sector_rotation","qqq_sts":qs,"spy_sts":ss,"strong_count":sn})
    return sigs

def recommend(mpc,sigs):
    lv=mpc["level"]; ha=any(s["type"]=="sector_alert" for s in sigs); hd=any(s["type"]=="panic_dip" for s in sigs)
    p=[]
    if lv=="extreme": p.append("极端恐慌 — 建议观望不参与")
    elif lv=="high": p.append("高恐慌 — 仓位<=30%,严格止损")
    elif lv=="elevated": p.append("恐慌升温"+(" — 关注错杀反弹" if hd else " — 待VIX见顶"))
    elif lv=="moderate": p.append("适度 — 按STS信号正常操作")
    else: p.append("安逸 — 顺势关注放量突破")
    if ha: p.append("AI板块集体走弱,收紧止损")
    if hd: p.append("关注错杀标的RSI回升确认")
    return "; ".join(p)

def generate_summary(rd,mpc,stocks,sigs,rec):
    E={"calm":"🟢","moderate":"🟡","elevated":"🟠","high":"🔴","extreme":"⚫"}
    L=[f"📊 **每日美股短线日报 | {rd}**\n",
       f"{E.get(mpc['level'],'⚪')} **市场恐慌度: {mpc['mpc']}/100 ({mpc['level']})**",
       f"VIX {mpc['components']['vix']['value']} | F&G {mpc['components']['fear_greed']['value']} (恐惧)"]
    if mpc.get("mpc_change") is not None:
        a="↑" if mpc["mpc_change"]>0 else "↓" if mpc["mpc_change"]<0 else "→"
        tag=" ⚡ 恐慌骤变" if abs(mpc["mpc_change"])>15 else (" 情绪稳定" if abs(mpc["mpc_change"])<=8 else "")
        L.append(f"较昨日 {mpc['mpc_change']:+} {a} —{tag}")
    L.append("")
    for sig in sigs:
        if sig["type"]=="sector_alert": L.append(f"⚠️ **AI板块集体预警**: {sig['count']}/8只MACD走弱")
        elif sig["type"]=="panic_dip": L.append(f"⚡ **恐慌错杀池**: {', '.join(sig['stocks'])}")
        elif sig["type"]=="momentum_breakout": L.append(f"🚀 **放量突破**: {', '.join(sig['stocks'])}")
        elif sig["type"]=="sector_rotation": L.append(f"🔥 **板块轮动**: {sig['strong_count']}只逆势走强")
    if sigs: L.append("")
    nonb=[s for s in stocks if s["ticker"] not in ("SPY","QQQ")]
    strong=sorted([s for s in nonb if s["sts"]>=55], key=lambda x: x["sts"], reverse=True)
    if strong:
        L.append("📈 **今日关注 (STS >=55):**")
        for s in strong:
            a=",".join(s["alerts"]) if s["alerts"] else ""; icon="🟢" if s["level"]=="strong" else "🟡"
            L.append(f"  {icon} **{s['ticker']}** ({s['sts']}) — {s['signals']['ma']['detail']}"+(f" ⚡{a}" if a else ""))
        L.append("")
    weak=[s for s in nonb if s["sts"]<30]
    if weak:
        L.append("👀 **弱势关注 (STS <30):**")
        for s in weak:
            r=s["signals"]["rsi"].get("value"); L.append(f"  🔴 **{s['ticker']}** ({s['sts']}) — RSI {r if r else 'N/A'}")
        L.append("")
    spy=next((s for s in stocks if s["ticker"]=="SPY"),None); qqq=next((s for s in stocks if s["ticker"]=="QQQ"),None)
    if spy and qqq: L.append(f"📉 **大盘基准**: SPY {spy['sts']} | QQQ {qqq['sts']}")
    L.append("")
    sb=[s for s in stocks if s["ticker"] not in ("SPY","QQQ")]
    sb_sorted=sorted(sb,key=lambda x: x["sts"],reverse=True)
    L.append("📋 **逐标的方向研判:**")
    for s in sb_sorted:
        L.append(generate_ticker_analysis(s))
        L.append(generate_prediction(s))
        sig=s["signals"]; rsi_v=sig["rsi"].get("raw")
        macd_l=sig["macd"].get("line"); macd_h=sig["macd"].get("histogram")
        bb_w=sig["bollinger"].get("bandwidth"); bb_u=sig["bollinger"].get("upper"); bb_l=sig["bollinger"].get("lower")
        ma20=sig["ma"].get("ma20"); ma50=sig["ma"].get("ma50")
        raw_parts=[]
        if rsi_v is not None: raw_parts.append(f"RSI {rsi_v}")
        if macd_l is not None: raw_parts.append(f"MACD {macd_l:.3f} (柱{macd_h:+.4f})" if macd_h else f"MACD {macd_l:.3f}")
        if bb_w is not None: raw_parts.append(f"布林带宽 {bb_w:.4f}")
        if ma20 and ma50: raw_parts.append(f"MA20 {ma20:.1f}/MA50 {ma50:.1f}")
        if raw_parts: L.append(f"  `{' | '.join(raw_parts)}`")
        L.append(generate_plain_explanation(s, mpc))
    L.append(f"\n⛔ **建议:** {rec}")
    return "\n".join(L)

BULLISH_WORDS = ["beat","surge","rally","upgrade","buy","outperform","raise","bull","jump","soar","record","growth","gain","boost","breakthrough","approval","launch","partnership","upside","beat expectations","raised guidance","strong demand"]
BEARISH_WORDS = ["fall","drop","miss","downgrade","sell","cut","bear","plunge","tumble","decline","loss","risk","warn","probe","investigation","lawsuit","delay","sanction","tariff","ban","recall","layoff","crash","concern","missed estimates","lowered guidance","weak demand"]

def classify_news_sentiment(title, summary=""):
    text = (title+" "+(summary or "")).lower()
    bh = sum(1 for w in BULLISH_WORDS if w in text)
    br = sum(1 for w in BEARISH_WORDS if w in text)
    if bh>br: return "bullish"
    elif br>bh: return "bearish"
    return "neutral"

def generate_news_summary(stock_news):
    if not stock_news: return ""
    lines=["📰 **最新消息:**"]
    for n in stock_news[:3]:
        s=classify_news_sentiment(n["title"],n.get("summary",""))
        icon={"bullish":"🐂 利好","bearish":"🐻 利空","neutral":"⚪"}[s]
        title=n["title"][:80]+("..." if len(n["title"])>80 else "")
        lines.append(f"  {icon} {title} ({n.get('source','')})")
    return "\n".join(lines)

def run_analysis(df_daily, df_history, vix, fng, prev_mpc=None, news_dict=None):
    if "avg_volume" not in df_daily.columns:
        df_daily=df_daily.copy(); df_daily["avg_volume"]=df_daily["volume"]
    mpc=compute_mpc(vix,fng,prev_mpc)
    # find QQQ for relative strength baseline
    qqq_row=df_daily[df_daily["ticker"]=="QQQ"]
    qqq_close=float(qqq_row["close"].iloc[0]) if len(qqq_row) else None
    qqq_hist=df_history["QQQ"].dropna().values if "QQQ" in df_history.columns else np.array([])
    stocks=[]
    for _,row in df_daily.iterrows():
        t=row["ticker"]; hist=df_history[t].dropna().values if t in df_history.columns else np.array([row["close"]]*30)
        s=compute_sts_one(t,row["close"],row.get("ma20",np.nan),row.get("ma50",np.nan),
                          row["volume"],row["avg_volume"],row.get("change_pct",0),hist,
                          row.get("target_price"),row.get("recommendation"))
        # fill relative strength
        if qqq_close and len(qqq_hist)>=5 and len(hist)>=5:
            rs5=round((row["close"]/hist[-5]-1)*100-(qqq_close/qqq_hist[-5]-1)*100,2)
            s["relative_strength"]["rs_5d"]=rs5
            s["relative_strength"]["vs_qqq"]="outperform" if rs5>2 else "underperform" if rs5<-2 else "in_line"
        s["name_cn"] = TICKER_NAMES.get(t, "")
        if news_dict and t in news_dict: s["news"] = news_dict[t]
        s["news_text"] = generate_news_summary(s.get("news",[]))
        s["explanation"] = generate_plain_explanation(s, mpc)
        stocks.append(s)
    sigs=detect_signals(mpc,stocks,df_daily); rec=recommend(mpc,sigs)
    today_str=date.today().isoformat(); summary=generate_summary(today_str,mpc,stocks,sigs,rec)
    return {"date":today_str,"market_panic":mpc,"stocks":stocks,"special_signals":sigs,"recommendation":rec,"summary_markdown":summary}

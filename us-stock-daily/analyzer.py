"""analyzer.py — 美股短线投机分析引擎 | Phase 2 · Data Analyst · 2026-06-09"""
import numpy as np, pandas as pd
from typing import Optional, Tuple
from datetime import date

def vix_to_score(vix): return 0 if vix<13 else 15 if vix<17 else 30 if vix<20 else 50 if vix<25 else 70 if vix<30 else 85 if vix<35 else 100

def compute_mpc(vix, fng, prev_mpc=None):
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

def compute_sts_one(t,c,ma20,ma50,vol,avol,chg,hist):
    rv=compute_rsi(hist); ml,ms,mh,mph=compute_macd(hist); bm,bu,bl,bw,bpw=compute_bollinger(hist)
    sc={}; sc["ma"],md=score_ma(c,ma20,ma50); sc["rsi"],rv2=score_rsi(rv)
    sc["macd"],mcd=score_macd(ml,ms,mh,mph); sc["vol"],vd=score_volume(vol,avol,chg)
    sc["bb"],bd=score_bollinger(c,bu,bl,bm,bw,bpw)
    vw=sum(w for k,w in WEIGHTS.items() if not pd.isna(sc[k]))
    sts=sum(sc[k]*WEIGHTS[k]/vw for k in WEIGHTS if not pd.isna(sc[k])) if vw else 50; sts=round(sts,1)
    lv="strong" if sts>=70 else "bullish" if sts>=55 else "neutral" if sts>=40 else "weak" if sts>=25 else "bearish"
    return {"ticker":t,"sts":sts,"level":lv,
            "signals":{"ma":{"score":sc["ma"],"detail":md},"rsi":{"score":sc["rsi"],"value":round(rv2,1) if rv2 else None},
                       "macd":{"score":sc["macd"],"detail":mcd},"volume":{"score":sc["vol"],"detail":vd},
                       "bollinger":{"score":sc["bb"],"detail":bd}},"alerts":[]}

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
    L.append(f"\n⛔ **建议:** {rec}")
    return "\n".join(L)

def run_analysis(df_daily, df_history, vix, fng, prev_mpc=None):
    if "avg_volume" not in df_daily.columns:
        df_daily=df_daily.copy(); df_daily["avg_volume"]=df_daily["volume"]
    mpc=compute_mpc(vix,fng,prev_mpc); stocks=[]
    for _,row in df_daily.iterrows():
        t=row["ticker"]; hist=df_history[t].dropna().values if t in df_history.columns else np.array([row["close"]]*30)
        s=compute_sts_one(t,row["close"],row.get("ma20",np.nan),row.get("ma50",np.nan),
                          row["volume"],row["avg_volume"],row.get("change_pct",0),hist)
        stocks.append(s)
    sigs=detect_signals(mpc,stocks,df_daily); rec=recommend(mpc,sigs)
    today_str=date.today().isoformat(); summary=generate_summary(today_str,mpc,stocks,sigs,rec)
    return {"date":today_str,"market_panic":mpc,"stocks":stocks,"special_signals":sigs,"recommendation":rec,"summary_markdown":summary}

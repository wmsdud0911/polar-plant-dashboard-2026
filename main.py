import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io


st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df


def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data
def load_data(env_file, growth_file):
    if env_file is not None:
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)
    else:
        env_df = pd.DataFrame()
    if growth_file is not None:
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)
    else:
        growth_df = pd.DataFrame()
    return env_df, growth_df


def plot_individual_bar(df, y_col, title, highlight_top=False):
    if y_col not in df.columns:
        return None
    data = df.dropna(subset=[y_col]).sort_values(y_col, ascending=False)
    if len(data) == 0:
        return None
    if "개체번호" in data.columns:
        x_col = "개체번호"
    else:
        x_col = data.index
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        title=title,
        labels={str(x_col): "개체", y_col: ""},
    )
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        xaxis_tickangle=-45,
        margin=dict(t=40, b=80),
        xaxis_title="개체",
        yaxis_title="",
    )
    if highlight_top and len(data) >= 4:
        n = len(data)
        top_n = max(1, int(np.ceil(n * 0.25)))
        colors = ["#2ecc71" if i < top_n else "#3498db" for i in range(n)]
        fig.update_traces(marker_color=colors)
    return fig


def plot_scatter_with_trend(df, x_col, y_col, title):
    if x_col not in df.columns or y_col not in df.columns:
        return None
    data = df[[x_col, y_col]].dropna()
    if len(data) < 2:
        return None
    r = data[x_col].corr(data[y_col])
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        title=f"{title} (r = {r:.2f})",
        labels={x_col: x_col, y_col: y_col},
    )
    slope, intercept = np.polyfit(data[x_col], data[y_col], 1)
    x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="추세선",
        )
    )
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(t=40, b=40),
    )
    return fig


st.sidebar.title("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

growth_metrics = st.sidebar.multiselect(
    "📊 생육 측정 항목 선택",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
)

env_df, growth_df = load_data(env_file, growth_file)

school_name = "우리 학교"
if env_file is not None:
    raw_name = env_file.name.rsplit(".csv", 1)[0]
    parts = raw_name.split("_")
    school_name = parts[0] if len(parts) > 1 else raw_name
elif growth_file is not None:
    raw_name = growth_file.name.rsplit(".csv", 1)[0]
    parts = raw_name.split("_")
    school_name = parts[0] if len(parts) > 1 else raw_name

st.title(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            f"""
**{school_name}**에서 재배한 극지식물의 개체별 생육 차이를 분석합니다.

- 극지식물은 열악한 환경에서도 생존하는 식물로, 개체마다 생육 상태가 다를 수 있습니다.
- 이 대시보드는 각 개체의 **생중량**, **잎 수**, **지상부·지하부 길이**를 비교하여
  어떤 개체가 잘 자랐는지, 생육 지표들 사이에 어떤 관계가 있는지 살펴봅니다.
- 환경 데이터(온도, 습도, pH, EC)와 생육 결과를 함께 확인할 수 있습니다.
"""
        )

        st.subheader("📋 데이터 요약")
        summary_rows = []

        if len(growth_df) > 0:
            if "개체번호" in growth_df.columns:
                summary_rows.append(
                    {"항목": "총 개체 수", "값": f"{growth_df['개체번호'].nunique()}개체"}
                )
            else:
                summary_rows.append({"항목": "총 개체 수", "값": f"{len(growth_df)}개체"})

            measure_cols = [
                c
                for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
                if c in growth_df.columns
            ]
            summary_rows.append(
                {"항목": "생육 측정 항목", "값": ", ".join(measure_cols) if measure_cols else "없음"}
            )
        else:
            summary_rows.append({"항목": "총 개체 수", "값": "데이터 없음"})
            summary_rows.append({"항목": "생육 측정 항목", "값": "데이터 없음"})

        if len(env_df) > 0:
            if "time" in env_df.columns:
                time_min = env_df["time"].min()
                time_max = env_df["time"].max()
                if pd.notna(time_min) and pd.notna(time_max):
                    summary_rows.append(
                        {
                            "항목": "측정 기간",
                            "값": f"{time_min.strftime('%Y-%m-%d')} ~ {time_max.strftime('%Y-%m-%d')}",
                        }
                    )
                else:
                    summary_rows.append({"항목": "측정 기간", "값": "기간 정보 없음"})
            else:
                summary_rows.append({"항목": "측정 기간", "값": "시간 열 없음"})

            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            summary_rows.append(
                {"항목": "환경 측정 항목", "값": ", ".join(env_cols) if env_cols else "없음"}
            )
        else:
            summary_rows.append({"항목": "측정 기간", "값": "데이터 없음"})
            summary_rows.append({"항목": "환경 측정 항목", "값": "데이터 없음"})

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("📊 주요 지표")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if len(growth_df) > 0:
                if "개체번호" in growth_df.columns:
                    st.metric("총 개체수", f"{growth_df['개체번호'].nunique()}")
                else:
                    st.metric("총 개체수", f"{len(growth_df)}")
            else:
                st.metric("총 개체수", "데이터 없음")

        with c2:
            if len(env_df) > 0 and "temperature" in env_df.columns:
                avg_temp = env_df["temperature"].dropna().mean()
                if pd.notna(avg_temp):
                    st.metric("평균 온도", f"{avg_temp:.1f}℃")
                else:
                    st.metric("평균 온도", "계산 불가")
            else:
                st.metric("평균 온도", "데이터 없음")

        with c3:
            if len(env_df) > 0 and "humidity" in env_df.columns:
                avg_hum = env_df["humidity"].dropna().mean()
                if pd.notna(avg_hum):
                    st.metric("평균 습도", f"{avg_hum:.1f}%")
                else:
                    st.metric("평균 습도", "계산 불가")
            else:
                st.metric("평균 습도", "데이터 없음")

        with c4:
            if len(growth_df) > 0 and "생중량" in growth_df.columns:
                avg_bio = growth_df["생중량"].dropna().mean()
                if pd.notna(avg_bio):
                    st.metric("평균 생중량", f"{avg_bio:.2f}g")
                else:
                    st.metric("평균 생중량", "계산 불가")
            else:
                st.metric("평균 생중량", "데이터 없음")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        if len(env_df) == 0:
            st.info("📁 사이드바에서 환경 CSV를 업로드해 주세요.")
        else:
            st.header("🌡️ 환경 데이터 요약")

            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]

            if env_num_cols:
                st.subheader("환경 지표 요약 (평균·최소·최대)")
                env_stats = env_df[env_num_cols].dropna()
                if len(env_stats) > 0:
                    stats_data = {
                        "지표": env_num_cols,
                        "평균": [env_stats[c].mean() for c in env_num_cols],
                        "최소": [env_stats[c].min() for c in env_num_cols],
                        "최대": [env_stats[c].max() for c in env_num_cols],
                    }
                    unit_map = {"temperature": "℃", "humidity": "%", "ph": "", "ec": "mS/cm"}
                    stats_data["단위"] = [unit_map.get(c, "") for c in env_num_cols]
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("환경 데이터에서 숫자 값을 찾을 수 없어요.")

            if "time" in env_df.columns:
                st.subheader("📈 날짜별 환경 변화 (일별 평균)")

                env_sorted = env_df.dropna(subset=["time"]).sort_values("time")
                env_sorted["date"] = env_sorted["time"].dt.date

                daily_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]

                if daily_cols:
                    daily = env_sorted.groupby("date")[daily_cols].mean().reset_index()

                    if "temperature" in daily_cols:
                        fig_temp = px.line(
                            daily,
                            x="date",
                            y="temperature",
                            title="일별 평균 온도 변화",
                            labels={"date": "날짜", "temperature": "온도(℃)"},
                        )
                        fig_temp.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            xaxis_tickangle=-45,
                        )
                        st.plotly_chart(fig_temp, width="stretch")

                    if "humidity" in daily_cols:
                        fig_hum = px.line(
                            daily,
                            x="date",
                            y="humidity",
                            title="일별 평균 습도 변화",
                            labels={"date": "날짜", "humidity": "습도(%)"},
                        )
                        fig_hum.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            xaxis_tickangle=-45,
                        )
                        st.plotly_chart(fig_hum, width="stretch")

                    ph_ec_cols = [c for c in ["ph", "ec"] if c in daily_cols]
                    if ph_ec_cols:
                        st.subheader("pH·EC 변화")
                        fig_pe = make_subplots(specs=[[{"secondary_y": False}]])
                        for col in ph_ec_cols:
                            fig_pe.add_trace(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily[col],
                                    mode="lines",
                                    name=col.upper(),
                                ),
                                secondary_y=False,
                            )
                        fig_pe.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            xaxis_tickangle=-45,
                            margin=dict(t=40, b=80),
                        )
                        st.plotly_chart(fig_pe, width="stretch")
                else:
                    st.info("환경 데이터에서 표시할 숫자 열이 없어요.")
            else:
                st.info("시간 열이 없어 날짜별 시계열을 표시할 수 없어요.")

            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                if len(env_df) > 0:
                    st.dataframe(env_df, use_container_width=True)
                    csv_data = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드 (utf-8-sig)",
                        data=csv_data,
                        file_name=f"{school_name}_환경데이터_다운로드.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 환경 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        if len(growth_df) == 0:
            st.info("📁 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.header("🥇 핵심 결과")

            c1, c2, c3 = st.columns(3)

            with c1:
                if "생중량" in growth_df.columns:
                    avg_bio = growth_df["생중량"].dropna().mean()
                    if pd.notna(avg_bio):
                        st.metric("평균 생중량", f"{avg_bio:.2f}g")
                    else:
                        st.metric("평균 생중량", "계산 불가")
                else:
                    st.metric("평균 생중량", "생중량 열 없음")

            with c2:
                if "생중량" in growth_df.columns:
                    valid_bio = growth_df["생중량"].dropna()
                    if len(valid_bio) > 0:
                        max_val = valid_bio.max()
                        max_idx = valid_bio.idxmax()
                        if "개체번호" in growth_df.columns:
                            max_id = growth_df.loc[max_idx, "개체번호"]
                        else:
                            max_id = f"#{max_idx}"
                        st.metric("최대 생중량", f"{max_val:.2f}g ({max_id})")
                    else:
                        st.metric("최대 생중량", "계산 불가")
                else:
                    st.metric("최대 생중량", "생중량 열 없음")

            with c3:
                if "생중량" in growth_df.columns:
                    valid_bio = growth_df["생중량"].dropna()
                    if len(valid_bio) > 0:
                        min_val = valid_bio.min()
                        min_idx = valid_bio.idxmin()
                        if "개체번호" in growth_df.columns:
                            min_id = growth_df.loc[min_idx, "개체번호"]
                        else:
                            min_id = f"#{min_idx}"
                        st.metric("최소 생중량", f"{min_val:.2f}g ({min_id})")
                    else:
                        st.metric("최소 생중량", "계산 불가")
                else:
                    st.metric("최소 생중량", "생중량 열 없음")

            st.subheader("📊 개체별 생육 비교")

            if "생중량" in growth_df.columns:
                st.markdown("#### 평균 생중량 (⭐ 가장 중요)")
                fig_bio = plot_individual_bar(
                    growth_df, "생중량", "개체별 생중량 비교", highlight_top=True
                )
                if fig_bio is not None:
                    st.plotly_chart(fig_bio, width="stretch")
                    n = growth_df["생중량"].dropna().shape[0]
                    top_n = max(1, int(np.ceil(n * 0.25)))
                    st.info(f"🟢 초록색 = 생중량 상위 25% ({top_n}개체), 🔵 파란색 = 나머지")

            for metric in growth_metrics:
                col_map = {
                    "생중량": "생중량",
                    "잎 수": "잎 수",
                    "지상부 길이": "지상부 길이",
                    "지하부 길이": "지하부 길이",
                }
                y_col = col_map.get(metric)
                if y_col and y_col in growth_df.columns and y_col != "생중량":
                    st.markdown(f"#### 평균 {metric}")
                    fig_m = plot_individual_bar(growth_df, y_col, f"개체별 {metric}")
                    if fig_m is not None:
                        st.plotly_chart(fig_m, width="stretch")

            if "생중량" in growth_df.columns:
                st.subheader("📦 생중량 분포")
                bio_data = growth_df["생중량"].dropna()
                if len(bio_data) >= 2:
                    fig_dist = px.box(
                        growth_df.dropna(subset=["생중량"]),
                        y="생중량",
                        title="생중량 분포 (박스플롯)",
                        labels={"생중량": "생중량(g)"},
                    )
                    fig_dist.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig_dist, width="stretch")

                    mean_b = bio_data.mean()
                    std_b = bio_data.std()
                    cv_b = (std_b / mean_b * 100) if mean_b > 0 else 0
                    d1, d2, d3 = st.columns(3)
                    d1.metric("평균", f"{mean_b:.2f}g")
                    d2.metric("표준편차", f"{std_b:.2f}g")
                    d3.metric("변동계수(CV)", f"{cv_b:.1f}%")
                    st.caption(
                        "※ 표준편차: 데이터가 평균에서 얼마나 퍼져 있는지 나타내는 값\n"
                        "※ CV(변동계수) = 표준편차 ÷ 평균 × 100, 값이 클수록 개체 간 차이가 큼"
                    )
                else:
                    st.info("생중량 데이터를 충분히 확보하지 못했어요.")

            st.subheader("🔗 상관관계 분석")

            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                st.markdown("#### 잎 수 vs 생중량")
                fig1 = plot_scatter_with_trend(growth_df, "잎 수", "생중량", "잎 수 vs 생중량")
                if fig1 is not None:
                    st.plotly_chart(fig1, width="stretch")
                    r1 = (
                        growth_df[["잎 수", "생중량"]]
                        .dropna()["잎 수"]
                        .corr(growth_df[["잎 수", "생중량"]].dropna()["생중량"])
                    )
                    st.caption(
                        f"※ 상관계수 r = {r1:.2f} — 1에 가까울수록 강한 양의 상관, "
                        f"-1에 가까울수록 강한 음의 상관, 0에 가까우면 상관 없음"
                    )
                else:
                    st.info("잎 수와 생중량 데이터를 충분히 확보하지 못했어요.")

            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                st.markdown("#### 지상부 길이 vs 생중량")
                fig2 = plot_scatter_with_trend(
                    growth_df, "지상부 길이", "생중량", "지상부 길이 vs 생중량"
                )
                if fig2 is not None:
                    st.plotly_chart(fig2, width="stretch")
                    r2 = (
                        growth_df[["지상부 길이", "생중량"]]
                        .dropna()["지상부 길이"]
                        .corr(growth_df[["지상부 길이", "생중량"]].dropna()["생중량"])
                    )
                    st.caption(
                        f"※ 상관계수 r = {r2:.2f} — 1에 가까울수록 강한 양의 상관, "
                        f"-1에 가까울수록 강한 음의 상관, 0에 가까우면 상관 없음"
                    )
                else:
                    st.info("지상부 길이와 생중량 데이터를 충분히 확보하지 못했어요.")

            if "생중량" in growth_df.columns:
                st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")
                bio_valid = growth_df.dropna(subset=["생중량"])
                if len(bio_valid) >= 4:
                    bio_sorted = bio_valid.sort_values("생중량")
                    n = len(bio_sorted)
                    q_n = max(1, int(np.ceil(n * 0.25)))

                    bottom = bio_sorted.head(q_n)
                    top = bio_sorted.tail(q_n)

                    comp_rows = []
                    for m_label, m_col in [
                        ("잎 수", "잎 수"),
                        ("지상부 길이", "지상부 길이"),
                        ("지하부 길이", "지하부 길이"),
                    ]:
                        if m_col in growth_df.columns:
                            b_mean = bottom[m_col].dropna().mean()
                            t_mean = top[m_col].dropna().mean()
                            comp_rows.append(
                                {
                                    "측정 항목": m_label,
                                    "하위 25% 평균": f"{b_mean:.2f}" if pd.notna(b_mean) else "N/A",
                                    "상위 25% 평균": f"{t_mean:.2f}" if pd.notna(t_mean) else "N/A",
                                    "차이": (
                                        f"{t_mean - b_mean:+.2f}"
                                        if pd.notna(b_mean) and pd.notna(t_mean)
                                        else "N/A"
                                    ),
                                }
                            )

                    if "생중량" in growth_df.columns:
                        b_bio = bottom["생중량"].dropna().mean()
                        t_bio = top["생중량"].dropna().mean()
                        comp_rows.append(
                            {
                                "측정 항목": "생중량",
                                "하위 25% 평균": f"{b_bio:.2f}g" if pd.notna(b_bio) else "N/A",
                                "상위 25% 평균": f"{t_bio:.2f}g" if pd.notna(t_bio) else "N/A",
                                "차이": (
                                    f"{t_bio - b_bio:+.2f}g"
                                    if pd.notna(b_bio) and pd.notna(t_bio)
                                    else "N/A"
                                ),
                            }
                        )

                    if comp_rows:
                        comp_df = pd.DataFrame(comp_rows)
                        st.dataframe(comp_df, use_container_width=True)
                    else:
                        st.info("비교할 측정 항목이 없어요.")
                else:
                    st.info("비교하려면 최소 4개체 이상의 데이터가 필요해요.")

            with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                if len(growth_df) > 0:
                    st.dataframe(growth_df, use_container_width=True)
                    csv_data = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드 (utf-8-sig)",
                        data=csv_data,
                        file_name=f"{school_name}_생육결과_다운로드.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 생육 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

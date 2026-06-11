import base64
import hashlib
import io
import json

import folium
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from streamlit_folium import st_folium

MAX_MESSAGES = 20


def geocode(place_name: str) -> tuple | None:
    """장소명 → (lat, lon, display_name). 실패 시 None."""
    try:
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place_name, "format": "json", "limit": 1},
            headers={"User-Agent": "travel-chatbot/1.0"},
            timeout=5,
        )
        data = res.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        pass
    return None


def extract_places(client: OpenAI, text: str) -> list[str]:
    """AI 응답 텍스트에서 여행 장소명 리스트를 영어로 추출."""
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract all specific travel locations (cities, landmarks, restaurants, "
                    "hotels, neighborhoods, attractions) from the text. "
                    "Return ONLY a JSON array of strings in English. "
                    "Example: [\"Tokyo\", \"Shibuya\", \"Tsukiji Market\"]. "
                    "If none found, return []."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    raw = res.choices[0].message.content.strip()
    try:
        places = json.loads(raw)
        if isinstance(places, list):
            return [p for p in places if isinstance(p, str)]
    except Exception:
        pass
    return []


def identify_place_from_image(client: OpenAI, image_bytes: bytes, mime_type: str) -> str | None:
    """이미지에서 장소명을 영어로 추출. 인식 불가 시 None."""
    b64 = base64.b64encode(image_bytes).decode()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "What specific place, restaurant, cafe, landmark, or shop is shown in this image? "
                            "Reply with ONLY the place name in English (e.g. 'Tsukiji Outer Market, Tokyo'). "
                            "If you cannot identify a specific location, reply with exactly: UNKNOWN"
                        ),
                    },
                ],
            }
        ],
        max_tokens=100,
        temperature=0,
    )
    name = res.choices[0].message.content.strip()
    return None if name == "UNKNOWN" else name


def generate_travel_plan(client: OpenAI, messages: list[dict]) -> str:
    """대화 내역을 바탕으로 구조화된 여행 계획서를 생성."""
    conversation = "\n".join(
        f"[{'사용자' if m['role'] == 'user' else 'AI'}] {m['content']}"
        for m in messages
        if m["role"] in ("user", "assistant")
    )
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 여행 플래너입니다. 아래 대화 내역을 분석해서 "
                    "사용자 맞춤 여행 계획서를 한국어 마크다운으로 작성하세요.\n\n"
                    "반드시 아래 구조를 따르세요:\n"
                    "# ✈️ 나의 여행 계획서\n"
                    "## 📌 여행 개요 (목적지·기간·예산·여행 스타일)\n"
                    "## 🗓️ 일별 일정 (Day 1 ~ Day N, 오전/오후/저녁 구분)\n"
                    "## 🍽️ 추천 맛집 & 카페\n"
                    "## 🏨 숙소 추천\n"
                    "## 🚌 교통 정보\n"
                    "## 💰 예산 계획 (항목별 예상 비용)\n"
                    "## 💡 여행 꿀팁\n\n"
                    "대화에서 언급되지 않은 항목은 대화 맥락을 바탕으로 적절히 채워주세요."
                ),
            },
            {"role": "user", "content": f"다음 대화를 바탕으로 여행 계획서를 작성해주세요:\n\n{conversation}"},
        ],
    )
    return res.choices[0].message.content


def build_map(locations: list[tuple]) -> folium.Map:
    """locations: [(lat, lon, display_name), ...] → folium.Map"""
    lats = [loc[0] for loc in locations]
    lons = [loc[1] for loc in locations]
    m = folium.Map(
        location=[sum(lats) / len(lats), sum(lons) / len(lons)],
        zoom_start=13 if len(locations) == 1 else 11,
    )
    for i, (lat, lon, display_name) in enumerate(locations, 1):
        short_name = display_name.split(",")[0]
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(display_name, max_width=260),
            tooltip=f"{i}. {short_name}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    return m


# ── 앱 ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* 헤더 배너 */
.travel-header {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 40%, #0277bd 100%);
    border-radius: 16px;
    padding: 32px 36px 28px;
    margin-bottom: 8px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.travel-header h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.travel-header p {
    font-size: 1rem;
    color: #b3d9f7;
    margin: 0;
}

/* 추천 질문 카드 버튼 */
div[data-testid="stButton"] > button.suggest-btn {
    background: #1A2744;
    border: 1px solid #2a3f6f;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: left;
    font-size: 0.9rem;
    color: #e0eeff;
    transition: all 0.2s;
    height: auto;
    white-space: normal;
    line-height: 1.5;
}
div[data-testid="stButton"] > button.suggest-btn:hover {
    background: #223060;
    border-color: #4FC3F7;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(79,195,247,0.2);
}
</style>

<div class="travel-header">
    <h1>🌏 여행 플래너</h1>
    <p>AI와 함께 나만의 완벽한 여행을 계획해보세요</p>
</div>
""", unsafe_allow_html=True)

TRAVEL_STYLES = {
    "🎨 힙쟁이": "트렌디한 카페, 감성 골목, 인스타 핫플, 로컬 편집샵 위주로 추천해줘.",
    "🍜 먹방러": "현지 맛집, 길거리 음식, 야시장, 숨은 로컬 식당 위주로 추천해줘.",
    "🏖️ 휴양러": "리조트, 해변, 스파, 풀빌라 등 힐링과 여유 위주로 추천해줘.",
    "🏛️ 문화탐방러": "박물관, 유적지, 전통시장, 현지 문화 체험 위주로 추천해줘.",
    "🛍️ 쇼핑러": "쇼핑몰, 로컬 마켓, 편집샵, 면세점 등 쇼핑 위주로 추천해줘.",
}

with st.sidebar:
    st.header("⚙️ 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.caption("API 키는 [여기](https://platform.openai.com/account/api-keys)서 발급받을 수 있습니다.")

    st.divider()

    st.markdown("**🧳 내 여행 스타일**")
    st.caption("나에게 맞는 스타일을 눌러보세요. (복수 선택 가능)")

    if "active_styles" not in st.session_state:
        st.session_state.active_styles = []

    for label in TRAVEL_STYLES:
        is_on = label in st.session_state.active_styles
        if st.button(
            f"{'✅ ' if is_on else ''}{label}",
            key=f"style_btn_{label}",
            use_container_width=True,
            type="primary" if is_on else "secondary",
        ):
            if is_on:
                st.session_state.active_styles.remove(label)
            else:
                st.session_state.active_styles.append(label)
            st.rerun()

    selected_styles = st.session_state.active_styles

    if selected_styles:
        style_desc = " ".join(TRAVEL_STYLES[s] for s in selected_styles)
        system_prompt = (
            f"당신은 친절하고 경험 많은 여행 플래너입니다. "
            f"사용자의 여행 스타일은 {', '.join(selected_styles)} 입니다. "
            f"{style_desc} "
            f"항상 한국어로 답변하며 실용적이고 구체적인 조언을 제공합니다."
        )
    else:
        system_prompt = (
            "당신은 친절하고 경험 많은 여행 플래너입니다. "
            "사용자의 여행 목적지, 일정, 예산, 관심사를 파악하여 "
            "맞춤형 여행 계획을 제안합니다. "
            "항상 한국어로 답변하며 실용적이고 구체적인 조언을 제공합니다."
        )

    st.divider()

    st.caption(f"메시지 수 제한: 최근 {MAX_MESSAGES}개")
    if "messages" in st.session_state:
        st.caption(f"현재 대화: {len(st.session_state.messages)}개")
    if "map_locations" in st.session_state:
        st.caption(f"지도 마커: {len(st.session_state.map_locations)}개")

    if st.button("📋 여행 계획서 만들기", use_container_width=True, type="primary"):
        st.session_state["_make_plan"] = True
        st.rerun()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.map_locations = []
        st.session_state.geocoded_names = set()
        st.session_state.pop("travel_plan", None)
        st.rerun()


if st.session_state.get("active_styles"):
    pills = " ".join(f"`{s}`" for s in st.session_state.active_styles)
    st.markdown(f"**현재 여행 스타일:** {pills}")
else:
    st.caption("왼쪽 사이드바에서 여행 스타일을 선택하면 맞춤 추천을 받을 수 있어요.")

if not openai_api_key:
    st.info("사이드바에 OpenAI API 키를 입력하면 여행 계획을 시작할 수 있습니다.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    WELCOME = (
        "안녕하세요! 저는 AI 여행 플래너입니다 ✈️\n\n"
        "목적지, 일정, 예산을 알려주시면 맞춤 여행 계획을 도와드립니다. "
        "아래 예시 중 하나를 골라보세요!\n\n"
        "- 🗼 **도쿄 3박 4일** 추천해줘 (예산 100만원)\n"
        "- 🏖️ **방콕 가성비** 여행 계획 짜줘\n"
        "- 🗺️ **유럽 2주** 일정 짜줘\n"
        "- 🍊 **제주도 당일치기** 코스 알려줘"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
    if "map_locations" not in st.session_state:
        st.session_state.map_locations = []
    if "geocoded_names" not in st.session_state:
        st.session_state.geocoded_names = set()
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "last_image_hash" not in st.session_state:
        st.session_state.last_image_hash = None
    if "travel_plan" not in st.session_state:
        st.session_state.travel_plan = None

    # ── 여행 계획서 생성 트리거 ───────────────────────────────────────────────
    if st.session_state.pop("_make_plan", False):
        user_msgs = [m for m in st.session_state.messages if m["role"] != "assistant" or m["content"] != st.session_state.messages[0]["content"]]
        if len(user_msgs) < 2:
            st.toast("대화를 조금 더 나눈 뒤 계획서를 만들어보세요!", icon="💬")
        else:
            with st.spinner("여행 계획서를 작성하는 중입니다..."):
                st.session_state.travel_plan = generate_travel_plan(client, st.session_state.messages)

    # ── 대화 출력 ────────────────────────────────────────────────────────────
    SUGGESTIONS = [
        ("🗼", "도쿄 3박 4일 추천해줘 (예산 100만원)"),
        ("🏖️", "방콕 가성비 여행 계획 짜줘"),
        ("🗺️", "유럽 2주 일정 짜줘"),
        ("🍊", "제주도 당일치기 코스 알려줘"),
    ]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 웰컴 메시지 다음에만 추천 칩 표시
    if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
        cols = st.columns(2)
        for i, (icon, query) in enumerate(SUGGESTIONS):
            if cols[i % 2].button(f"{icon} {query}", key=f"sug_{i}", use_container_width=True):
                st.session_state["_suggested"] = query
                st.rerun()

    # ── 지도 출력 (채팅 답변 아래) ───────────────────────────────────────────
    if st.session_state.map_locations:
        with st.chat_message("assistant"):
            st.subheader("🗺️ 추천 장소 지도")
            st_folium(
                build_map(st.session_state.map_locations),
                use_container_width=True,
                height=420,
                returned_objects=[],
            )

    # ── 여행 계획서 표시 ──────────────────────────────────────────────────────
    if st.session_state.travel_plan:
        st.divider()
        st.markdown(st.session_state.travel_plan)
        st.download_button(
            label="📥 계획서 다운로드 (.md)",
            data=st.session_state.travel_plan,
            file_name="여행계획서.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("🔄 계획서 다시 생성", use_container_width=True):
            st.session_state["_make_plan"] = True
            st.session_state.travel_plan = None
            st.rerun()

    # ── 입력 영역 ────────────────────────────────────────────────────────────
    with st.expander("📷 이미지로 장소 찾기"):
        uploaded_file = st.file_uploader(
            "가게·명소 사진을 올리면 지도에 자동으로 표시해드립니다",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )
        if uploaded_file:
            image_bytes = uploaded_file.read()
            image_hash = hashlib.md5(image_bytes).hexdigest()
            if image_hash != st.session_state.last_image_hash:
                st.session_state.last_image_hash = image_hash
                col_img, col_result = st.columns([1, 2])
                with col_img:
                    st.image(image_bytes, use_container_width=True)
                with col_result:
                    with st.spinner("이미지에서 장소를 인식하는 중..."):
                        place_name = identify_place_from_image(client, image_bytes, uploaded_file.type)
                    if place_name:
                        result = geocode(place_name)
                        if result and place_name not in st.session_state.geocoded_names:
                            st.session_state.geocoded_names.add(place_name)
                            st.session_state.map_locations.append(result)
                            st.success(f"📍 지도에 추가됨: **{result[2].split(',')[0]}**")
                            msg = f"이미지에서 **{place_name}** 를 인식했습니다. 지도에 표시했습니다."
                        elif result:
                            st.info(f"이미 지도에 표시된 장소입니다: {result[2].split(',')[0]}")
                            msg = None
                        else:
                            st.warning(f"'{place_name}' 의 좌표를 찾지 못했습니다.")
                            msg = None
                    else:
                        st.warning("장소를 인식하지 못했습니다. 간판이 잘 보이는 사진을 시도해보세요.")
                        msg = None
                    if msg:
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        st.rerun()

    col_mic, col_label = st.columns([1, 9])
    with col_mic:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e87070",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
        )
    with col_label:
        st.caption("🎙️ 말하거나 아래 입력창에 직접 입력하세요.")

    voice_prompt = None
    if audio_bytes:
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("음성을 텍스트로 변환하는 중..."):
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.wav"
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",
                )
            voice_prompt = transcript.text
            st.toast(f"인식된 텍스트: {voice_prompt}", icon="🎙️")

    # ── 입력 처리 (추천 버튼 / 음성 / 텍스트) ─────────────────────────────────
    typed_prompt = st.chat_input("예: 도쿄 3박 4일 여행 계획 짜줘 (예산 100만원)")
    suggested_prompt = st.session_state.pop("_suggested", None)
    prompt = suggested_prompt or voice_prompt or typed_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        recent = st.session_state.messages[-MAX_MESSAGES:]

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}]
            + [{"role": m["role"], "content": m["content"]} for m in recent],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if len(st.session_state.messages) > MAX_MESSAGES:
            st.toast(
                f"대화가 {MAX_MESSAGES}개를 넘었습니다. 오래된 메시지는 AI 컨텍스트에서 제외됩니다.",
                icon="⚠️",
            )

        # ── 3. 장소 추출 — 스피너 없이 조용히 처리, 완료 시 toast ──────────────
        new_places = extract_places(client, response)
        added = 0
        for place in new_places:
            if place in st.session_state.geocoded_names:
                continue
            result = geocode(place)
            st.session_state.geocoded_names.add(place)
            if result:
                st.session_state.map_locations.append(result)
                added += 1

        if added:
            st.toast(f"지도에 {added}개 장소를 추가했습니다.", icon="📍")
            st.rerun()

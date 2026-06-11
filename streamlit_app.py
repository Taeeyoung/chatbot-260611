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

st.title("✈️ 여행 플래너 챗봇")

with st.sidebar:
    st.header("⚙️ 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.caption("API 키는 [여기](https://platform.openai.com/account/api-keys)서 발급받을 수 있습니다.")

    st.divider()

    system_prompt = st.text_area(
        "System Prompt",
        value=(
            "당신은 친절하고 경험 많은 여행 플래너입니다. "
            "사용자의 여행 목적지, 일정, 예산, 관심사를 파악하여 "
            "맞춤형 여행 계획, 관광지 추천, 현지 음식, 교통편, 숙소 정보를 제공합니다. "
            "항상 한국어로 답변하며, 실용적이고 구체적인 조언을 제공합니다."
        ),
        height=160,
        help="챗봇의 역할과 성격을 정의합니다.",
    )

    st.divider()

    st.caption(f"메시지 수 제한: 최근 {MAX_MESSAGES}개")
    if "messages" in st.session_state:
        st.caption(f"현재 대화: {len(st.session_state.messages)}개")
    if "map_locations" in st.session_state:
        st.caption(f"지도 마커: {len(st.session_state.map_locations)}개")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.map_locations = []
        st.session_state.geocoded_names = set()
        st.rerun()

st.caption("목적지, 예산, 일정을 알려주시면 맞춤 여행 계획을 도와드립니다 🗺️")

if not openai_api_key:
    st.info("사이드바에 OpenAI API 키를 입력하면 여행 계획을 시작할 수 있습니다.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "map_locations" not in st.session_state:
        st.session_state.map_locations = []
    if "geocoded_names" not in st.session_state:
        st.session_state.geocoded_names = set()
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    if "last_image_hash" not in st.session_state:
        st.session_state.last_image_hash = None

    # ── 2. 사이드바 하단에 지도 표시 ─────────────────────────────────────────
    with st.sidebar:
        if st.session_state.map_locations:
            st.divider()
            st.subheader("🗺️ 추천 장소 지도")
            st_folium(
                build_map(st.session_state.map_locations),
                use_container_width=True,
                height=320,
                returned_objects=[],
            )

    # ── 대화 출력 ────────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── 1. 빈 화면 — 추천 질문 버튼 ──────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**어떤 여행을 계획하고 계신가요? 아래 예시를 클릭해보세요.**")
        suggestions = [
            "도쿄 3박 4일 추천해줘 (예산 100만원)",
            "방콕 가성비 여행 계획 짜줘",
            "유럽 2주 일정 짜줘",
            "제주도 당일치기 코스 알려줘",
        ]
        col1, col2 = st.columns(2)
        for i, s in enumerate(suggestions):
            if (col1 if i % 2 == 0 else col2).button(s, use_container_width=True):
                st.session_state["_suggested"] = s
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

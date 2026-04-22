import streamlit as st

from camera_map import get_nearby_cameras
from camera_workers import get_camera_display_name


def render_camera_stats(stats, camera_points, key_prefix="stats"):
    st.markdown("### Camera Stats")

    with st.container(border=True):
        metric_row_1 = st.columns(4)
        metric_row_1[0].metric("Vehicle Count", int(stats.get("vehicle_count", 0)))
        metric_row_1[1].metric("Traffic Score (1-10)", f"{float(stats.get('traffic_score', 0)):.2f}")
        metric_row_1[2].metric("Traffic Rating", stats.get("traffic_label", "-"))
        metric_row_1[3].metric("Coverage (Cars % on Road)", f"{float(stats.get('coverage', 0)):.2f}%")

    with st.container(border=True):
        movement_counts = stats.get("movement_counts", {})
        metric_row_2 = st.columns(3)
        metric_row_2[0].metric("Stopped Cars", int(movement_counts.get("stopped", 0)))
        metric_row_2[1].metric("Slow Cars", int(movement_counts.get("slow", 0)))
        metric_row_2[2].metric("Fast Cars", int(movement_counts.get("fast", 0)))

    with st.container(border=True):
        detail_col_1, detail_col_2 = st.columns(2)
        with detail_col_1:
            raw_stream_url = stats.get("raw_stream_url", "")
            if isinstance(raw_stream_url, str) and raw_stream_url.startswith(("http://", "https://")):
                st.markdown(f"**Raw Stream:** [Open Stream]({raw_stream_url})")
            else:
                st.markdown("**Raw Stream:** No stream")

            st.markdown(f"**Last Updated:** {stats.get('last_updated', '-') or '-'}")
            st.markdown(f"**Current Road Mask Percentage:** {float(stats.get('road_mask_percent', 0)):.2f}%")
            st.markdown(f"**FPS:** {float(stats.get('fps', 0)):.2f}")
            st.markdown(f"**Resolution:** {stats.get('resolution', '-') or '-'}")

        with detail_col_2:
            st.markdown("**Detected Vehicle Types**")
            class_counts = stats.get("class_counts", {})
            if not class_counts:
                st.caption("No vehicles detected")
            else:
                for name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
                    noun = f"{name}es" if name == "bus" else (name if count == 1 else f"{name}s")
                    st.write(f"{count} {noun}")

    selected_camera = stats.get("selected_camera")
    nearby = get_nearby_cameras(camera_points, selected_camera)
    selected_from_nearby = None
    with st.container(border=True):
        st.markdown("**Nearby Cameras (to selected camera)**")
        if not nearby:
            st.caption("Nearby cameras are unavailable yet.")
        else:
            for idx, item in enumerate(nearby):
                camera_name = item["camera"]["location"]
                label = f"{get_camera_display_name(camera_name)} ({item['direction']}, {item['miles']:.1f} mi)"
                if st.button(label, key=f"{key_prefix}_nearby_{idx}_{camera_name}", width="stretch"):
                    selected_from_nearby = camera_name

    return selected_from_nearby
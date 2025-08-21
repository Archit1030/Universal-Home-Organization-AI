// src/components/LiveVideo.jsx
import React from "react";

export default function LiveVideo() {
  // If your Flask backend serves MJPEG at /video_feed:
  const videoUrl = "http://localhost:5000/video_feed";

  // For local dev we show the stream; if you used a file, replace with path.
  return (
    <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <img src={videoUrl} alt="live" style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: 8 }} />
    </div>
  );
}

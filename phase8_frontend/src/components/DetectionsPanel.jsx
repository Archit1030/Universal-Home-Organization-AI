import React, { useEffect, useState } from "react";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Divider from "@mui/material/Divider";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

function groupByCategory(objects = []) {
  const groups = {};
  objects.forEach((o) => {
    const cat = o.category || "unknown";
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(o);
  });
  return groups;
}

export default function DetectionsPanel() {
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    let mounted = true;
    async function pollLoop() {
      while (mounted) {
        try {
          const res = await fetch("http://localhost:5000/detections");
          const data = await res.json();
          setDetections(data.objects || []);
        } catch (err) {
          console.error("Fetch error:", err);
        }
        await new Promise((r) => setTimeout(r, 700));
      }
    }
    pollLoop();
    return () => (mounted = false);
  }, []);

  const groups = groupByCategory(detections);

  return (
    <Box sx={{ mt: 1 }}>
      {Object.keys(groups).length === 0 && (
        <Typography variant="body2" color="text.secondary">No detections yet</Typography>
      )}

      {Object.entries(groups).map(([category, items]) => (
        <Box key={category} sx={{ mb: 2 }}>
          <Typography variant="subtitle1" sx={{ mb: 0.5 }}>
            {category} ({items.length})
          </Typography>
          <List dense disablePadding>
            {items.map((it, i) => (
              <React.Fragment key={i}>
                <ListItem>
                  <ListItemText
                    primary={it.name}
                    secondary={`zone: ${it.zone} • conf: ${it.confidence ?? "—"}`}
                  />
                </ListItem>
                <Divider />
              </React.Fragment>
            ))}
          </List>
        </Box>
      ))}
    </Box>
  );
}

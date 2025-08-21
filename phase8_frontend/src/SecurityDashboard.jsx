// src/SecurityDashboard.jsx
import React, { useEffect, useState, useRef } from "react";
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Avatar,
  Chip,
  Button,
  Stack,
  Divider,
  Paper,
  CircularProgress,
  IconButton,
} from "@mui/material";

// MUI icons (all valid names)
import DashboardIcon from "@mui/icons-material/Dashboard";
import HistoryIcon from "@mui/icons-material/History";
import SettingsIcon from "@mui/icons-material/Settings";
import InfoIcon from "@mui/icons-material/Info";
import SecurityIcon from "@mui/icons-material/Security";
import VisibilityIcon from "@mui/icons-material/Visibility";
import PeopleIcon from "@mui/icons-material/People";
import DirectionsCarIcon from "@mui/icons-material/DirectionsCar";
import LocalShippingIcon from "@mui/icons-material/LocalShipping";
import PetsIcon from "@mui/icons-material/Pets";
import BoltIcon from "@mui/icons-material/Bolt";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import RefreshIcon from "@mui/icons-material/Refresh";

// Backend base URL from Vite env
const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";
const DRAWER_WIDTH = 260;

function prettyConfidenceColor(conf) {
  if (conf >= 90) return "#4caf50";
  if (conf >= 75) return "#f59e0b";
  return "#fb923c";
}

function priorityChipStyles(priority) {
  const p = (priority || "").toLowerCase();
  if (p === "high") return { bgcolor: "rgba(244,67,54,0.12)", color: "#ef5350" };
  if (p === "medium") return { bgcolor: "rgba(255,193,7,0.08)", color: "#ffb300" };
  return { bgcolor: "rgba(76,175,80,0.08)", color: "#66bb6a" };
}

function IconForCategory(category) {
  if (!category) return <BoltIcon />;
  const s = category.toLowerCase();
  if (s.includes("person") || s.includes("people")) return <PeopleIcon />;
  if (s.includes("vehicle") || s.includes("car")) return <DirectionsCarIcon />;
  if (s.includes("package") || s.includes("parcel")) return <LocalShippingIcon />;
  if (s.includes("animal") || s.includes("pet")) return <PetsIcon />;
  return <BoltIcon />;
}

function AppSidebar() {
  const menuItems = [
    { title: "Dashboard", icon: DashboardIcon, isActive: true },
    { title: "History", icon: HistoryIcon, isActive: false },
    { title: "Settings", icon: SettingsIcon, isActive: false },
    { title: "About", icon: InfoIcon, isActive: false },
  ];

  return (
    <Drawer
      variant="permanent"
      open
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: DRAWER_WIDTH,
          boxSizing: "border-box",
          bgcolor: "#071023",
          color: "#e6eef8",
          borderRight: "1px solid #16202a",
        },
      }}
    >
      <Toolbar sx={{ px: 2, py: 1 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <Box sx={{ width: 44, height: 44, bgcolor: "#0369a1", borderRadius: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <SecurityIcon sx={{ color: "#fff" }} />
          </Box>
          <Box>
            <Typography sx={{ fontWeight: 700, fontSize: 16 }}>Universal Home AI</Typography>
            <Typography variant="caption" sx={{ color: "gray" }}>Security Monitor</Typography>
          </Box>
        </Stack>
      </Toolbar>

      <Box sx={{ px: 1, mt: 2 }}>
        <List>
          {menuItems.map((item) => (
            <ListItemButton key={item.title} sx={{ mb: 1, borderRadius: 1, "&:hover": { bgcolor: "#0c1720" } }}>
              <ListItemIcon sx={{ color: "inherit", minWidth: 36 }}><item.icon /></ListItemIcon>
              <ListItemText primary={item.title} />
            </ListItemButton>
          ))}
        </List>
      </Box>
    </Drawer>
  );
}

function LiveVideoCard() {
  const streamUrl = `${BACKEND}/video_feed`;
  const placeholder = "/placeholder-camera.png";

  return (
    <Card sx={{ bgcolor: "#0b1220", color: "#e6eef8", border: "1px solid #16202a" }}>
      <CardHeader
        title={
          <Stack direction="row" alignItems="center" spacing={1}>
            <VisibilityIcon sx={{ color: "#60a5fa" }} />
            <Typography variant="h6" sx={{ color: "white", fontSize: 16 }}>Live Video Feed</Typography>
            <Chip label="LIVE" size="small" sx={{ ml: "auto", bgcolor: "rgba(16,185,129,0.12)", color: "#4ade80", border: "1px solid rgba(16,185,129,0.25)" }} />
          </Stack>
        }
        sx={{ pb: 0 }}
      />
      <CardContent sx={{ p: 0 }}>
        <Box sx={{
          aspectRatio: "16/9",
          bgcolor: "#000",
          border: "1px solid #16202a",
          m: 2,
          borderRadius: 2,
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <img
            src={streamUrl}
            alt="live feed"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = placeholder; e.currentTarget.style.objectFit = "contain"; }}
          />
        </Box>
      </CardContent>
      <CardActions sx={{ px: 2, pb: 2 }}>
        <Typography variant="caption" sx={{ color: "#ccc" }}>Camera: Front Door — Camera 01</Typography>
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="outlined" sx={{ color: "white", borderColor: "#23303a" }}>Settings</Button>
      </CardActions>
    </Card>
  );
}

function StatsCards() {
  return (
    <Grid container spacing={1} sx={{ mb: 2 }}>
      <Grid item xs={6}>
        <Card sx={{ bgcolor: "#071023", border: "1px solid #16202a" }}>
          <CardContent>
            <Stack direction="row" spacing={2} alignItems="center">
              <Box sx={{ p: 1, bgcolor: "rgba(59,130,246,0.12)", borderRadius: 1 }}>
                <TrendingUpIcon sx={{ color: "#60a5fa" }} />
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 700, color: "#fff" }}>24</Typography>
                <Typography variant="caption" sx={{ color: "#ccc" }}>Today's Events</Typography>
              </Box>
            </Stack>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={6}>
        <Card sx={{ bgcolor: "#071023", border: "1px solid #16202a" }}>
          <CardContent>
            <Stack direction="row" spacing={2} alignItems="center">
              <Box sx={{ p: 1, bgcolor: "rgba(249,115,22,0.12)", borderRadius: 1 }}>
                <WarningAmberIcon sx={{ color: "#fb923c" }} />
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 700, color: "#fff" }}>4</Typography>
                <Typography variant="caption" sx={{ color: "#ccc" }}>Active Alerts</Typography>
              </Box>
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}

function QuickActions({ onClear, onDismiss }) {
  return (
    <Card sx={{ bgcolor: "#071023", border: "1px solid #16202a", mb: 2 }}>
      <CardHeader title={<Typography variant="subtitle2" sx={{ color: "white" }}><BoltIcon sx={{ mr: 1 }} /> Quick Actions</Typography>} sx={{ pb: 0 }} />
      <CardContent sx={{ pt: 1 }}>
        <Grid container spacing={1}>
          <Grid item xs={6}>
            <Button variant="outlined" fullWidth startIcon={<CheckCircleIcon />} onClick={onClear} sx={{ color: "#e6eef8", borderColor: "#23303a" }}>Clear All</Button>
          </Grid>
          <Grid item xs={6}>
            <Button variant="outlined" fullWidth startIcon={<CancelIcon />} onClick={onDismiss} sx={{ color: "#e6eef8", borderColor: "#23303a" }}>Dismiss</Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

function LiveDetections() {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    async function fetchDetections() {
      try {
        const res = await fetch(`${BACKEND}/detections`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const objs = Array.isArray(data.objects) ? data.objects : [];
        const mapped = objs.map((o, i) => ({
          id: o.id ?? i + 1,
          name: o.name ?? (o.category ?? "object"),
          category: (o.category ?? o.name ?? "unknown"),
          zone: o.zone ?? "unknown",
          confidence: o.confidence ?? Math.floor(Math.random() * 20) + 80,
          timestamp: o.timestamp ?? data.timestamp ?? new Date().toLocaleString(),
          status: o.status ?? "active",
          priority: o.priority ?? "medium",
          bbox: o.bbox ?? null,
        }));
        if (mountedRef.current) setDetections(mapped);
      } catch (err) {
        console.error("fetchDetections error:", err);
        if (mountedRef.current) setDetections([]);
      } finally {
        if (mountedRef.current) setLoading(false);
      }
    }
    fetchDetections();
    const id = setInterval(fetchDetections, 4000);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, []);

  const handleClearAll = () => console.log("Clear all clicked");
  const handleDismiss = () => console.log("Dismiss clicked");

  return (
    <Box>
      <StatsCards />
      <QuickActions onClear={handleClearAll} onDismiss={handleDismiss} />

      <Card sx={{ bgcolor: "#071023", border: "1px solid #16202a" }}>
        <CardHeader
          title={
            <Stack direction="row" alignItems="center" spacing={1}>
              <WarningAmberIcon sx={{ color: "#fb923c" }} />
              <Typography variant="h6" sx={{ color: "white", fontSize: 16 }}>Live Detections</Typography>
              <Chip
                label={`${detections.filter((d) => d.status === "active").length} Active`}
                size="small"
                sx={{ ml: "auto", bgcolor: "rgba(251,146,60,0.12)", color: "#fb923c", border: "1px solid rgba(251,146,60,0.25)" }}
              />
            </Stack>
          }
          sx={{ pb: 0 }}
        />

        <CardContent sx={{ p: 0 }}>
          <Box sx={{ maxHeight: 480, overflow: "auto", px: 2, py: 1 }}>
            {loading ? (
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", py: 6 }}>
                <CircularProgress />
              </Box>
            ) : detections.length === 0 ? (
              <Typography sx={{ color: "#ccc", p: 2 }}>No detections</Typography>
            ) : (
              <Stack spacing={1}>
                {detections.map((d, idx) => (
                  <Box key={d.id ?? idx}>
                    <Box sx={{
                      display: "flex",
                      alignItems: "flex-start",
                      justifyContent: "space-between",
                      p: 1.25,
                      borderRadius: 1,
                      bgcolor: "#071220",
                      border: "1px solid #112233",
                      "&:hover": { boxShadow: 2 }
                    }}>
                      <Stack direction="row" spacing={2} alignItems="flex-start" sx={{ minWidth: 0 }}>
                        <Avatar sx={{ bgcolor: "#0b1220", width: 44, height: 44 }}>{IconForCategory(d.category)}</Avatar>

                        <Box sx={{ minWidth: 0 }}>
                          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                            <Typography sx={{ fontWeight: 600, color: "white" }}>{d.name}</Typography>
                            <Chip label={d.priority} size="small" sx={{ ...priorityChipStyles(d.priority), ml: 1 }} />
                            <Box sx={{ ml: 1 }}>
                              <Box component="span" sx={{ width: 8, height: 8, borderRadius: "50%", display: "inline-block", background: d.status === "active" ? "#fb923c" : "#6b7280", ml: 1 }} />
                            </Box>
                          </Stack>
                          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                            <LocationOnIcon sx={{ fontSize: 14, color: "#9ca3af" }} />
                            <Typography variant="body2" sx={{ color: "#ccc" }}>{d.zone}</Typography>
                          </Stack>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <AccessTimeIcon sx={{ fontSize: 14, color: "#9ca3af" }} />
                            <Typography variant="body2" sx={{ color: "#ccc" }}>{d.timestamp}</Typography>
                          </Stack>
                        </Box>
                      </Stack>
                      <Box sx={{ textAlign: "right", ml: 2 }}>
                        <Typography sx={{ fontFamily: "monospace", fontWeight: 700, color: prettyConfidenceColor(d.confidence), mb: 0.5 }}>{Math.round(d.confidence)}%</Typography>
                        <Stack direction="row" spacing={0.5} alignItems="center" justifyContent="flex-end">
                          <TrendingUpIcon sx={{ color: "#9ca3af", fontSize: 14 }} />
                          <Typography variant="caption" sx={{ color: "#ccc", textTransform: "capitalize" }}>{d.status}</Typography>
                        </Stack>
                      </Box>
                    </Box>
                    {idx < detections.length - 1 && <Divider sx={{ borderColor: "#0b1b2a", my: 1 }} />}
                  </Box>
                ))}
              </Stack>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}

export default function SecurityDashboard() {
  return (
    <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "#050708" }}>
      <AppSidebar />
      <Box sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <AppBar position="static" sx={{ bgcolor: "#071220", borderBottom: "1px solid #112233" }}>
          <Toolbar sx={{ minHeight: 64 }}>
            <Typography variant="h6" sx={{ color: "#ccc" }}>Security Dashboard</Typography>
            <Box sx={{ ml: "auto", display: "flex", gap: 2, alignItems: "center" }}>
              <Chip label="System Online" variant="outlined" sx={{ borderColor: "rgba(16,185,129,0.2)", color: "#bbf7d0" }} />
              <IconButton color="inherit" onClick={() => window.location.reload()}>
                <RefreshIcon />
              </IconButton>
            </Box>
          </Toolbar>
        </AppBar>

        <Box sx={{ flex: 1, p: 3 }}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <LiveVideoCard />
            </Grid>
            <Grid item xs={12} md={4}>
              <LiveDetections />
            </Grid>
          </Grid>
        </Box>
      </Box>
    </Box>
  );
}

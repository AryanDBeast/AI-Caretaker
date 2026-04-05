"use client";

import { useState, useEffect } from "react";

// ── types ────────────────────────────────────────────────────────────────────
interface Caregiver {
  id: number;
  first_name: string;
  last_name: string;
}

interface Patient {
  id: number;
  first_name: string;
  last_name: string;
}

interface LogEntry {
  speaker: string;
  statement: string;
  timestamp: string;
  activity_type?: string;
  medication_type?: string;
  emergency_type?: string;
  severity_score?: number;
}

// ── helpers ───────────────────────────────────────────────────────────────────
function normalizeLog(raw: any[]): LogEntry {
  return {
    speaker:         raw[0],
    statement:       raw[1],
    timestamp:       raw[2],
    activity_type:   raw[3] ?? undefined,
    medication_type: raw[4] ?? undefined,
    emergency_type:  raw[5] ?? undefined,
    severity_score:  raw[6] ?? undefined,
  };
}

function fmtDate(ts: string) {
  try {
    const d = new Date(ts);
    return d.toLocaleDateString("en-US", { month:"2-digit", day:"2-digit", year:"2-digit" });
  } catch { return ts; }
}
function fmtTime(ts: string) {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false });
  } catch { return ""; }
}
function getInitials(firstName: string, lastName: string) {
  return `${firstName?.[0] ?? ""}${lastName?.[0] ?? ""}`.toUpperCase();
}

// ── brand colors ──────────────────────────────────────────────────────────────
const TEAL       = "#0d9488";
const TEAL_LIGHT = "#ccfbf1";
const TEAL_DARK  = "#0f766e";

// ── icons ─────────────────────────────────────────────────────────────────────
const Icons = {
  home:     <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z"/><path d="M9 21V12h6v9"/></svg>,
  chat:     <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>,
  alert:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
  pill:     <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M10.5 20H4a2 2 0 01-2-2V6a2 2 0 012-2h16a2 2 0 012 2v7"/><path d="M12 12H2"/><path d="M18.5 14a2.5 2.5 0 015 0v5.5a2.5 2.5 0 01-5 0V14z"/><path d="M21 16.5h-5"/></svg>,
  activity: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
  logout:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>,
  refresh:  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>,
  chevron:  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polyline points="9 18 15 12 9 6"/></svg>,
  menu:     <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>,
  search:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>,
  download: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  privacy:  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  terms:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
};

// ══════════════════════════════════════════════════════════════════════════════
export default function Home() {
  const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "";

  const [step, setStep]                           = useState<"caregiver"|"patient"|"dashboard">("caregiver");
  const [caregivers, setCaregivers]               = useState<Caregiver[]>([]);
  const [caregiversLoading, setCaregiversLoading] = useState(false);
  const [selectedCaregiver, setSelectedCaregiver] = useState<Caregiver | null>(null);
  const [patients, setPatients]                   = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient]     = useState<Patient | null>(null);
  const [logs, setLogs]                           = useState<LogEntry[]>([]);
  const [page, setPage]                           = useState("home");
  const [sidebarOpen, setSidebarOpen]             = useState(true);
  const [search, setSearch]                       = useState("");
  const [loading, setLoading]                     = useState(false);

  // fetch caregivers on mount — replaces hardcoded list
  useEffect(() => {
    setCaregiversLoading(true);
    fetch(`${BACKEND}/caregivers`)
      .then(r => r.json())
      .then((data: Caregiver[]) => setCaregivers(data))
      .catch(console.error)
      .finally(() => setCaregiversLoading(false));
  }, []);

  // fetch patients when caregiver is selected
  useEffect(() => {
    if (!selectedCaregiver) return;
    fetch(`${BACKEND}/patients`)
      .then(r => r.json())
      .then((data: Patient[]) => setPatients(data))
      .catch(console.error);
  }, [selectedCaregiver]);

  // fetch logs when patient is selected
  useEffect(() => {
    if (!selectedPatient) return;
    setLoading(true);
    fetch(`${BACKEND}/patients/${selectedPatient.id}/statements`)
      .then(r => r.json())
      .then(data => setLogs((data as any[]).map(normalizeLog)))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedPatient]);

  const refreshLogs = () => {
    if (!selectedPatient) return;
    setLoading(true);
    fetch(`${BACKEND}/patients/${selectedPatient.id}/statements`)
      .then(r => r.json())
      .then(data => setLogs((data as any[]).map(normalizeLog)))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const emergencies = logs.filter(l => l.emergency_type);
  const medicines   = logs.filter(l => l.medication_type);
  const activities  = logs.filter(l => l.activity_type);

  // ── STEP 1: caregiver select ─────────────────────────────────────────────
  if (step === "caregiver") {
    return (
      <div style={{ fontFamily:"'DM Sans', system-ui, sans-serif", background:"#f0fdf9", minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center" }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
          * { box-sizing: border-box; }
          .cg-btn { transition: all 0.18s ease; }
          .cg-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(13,148,136,0.25); }
        `}</style>
        <div style={{ position:"fixed", top:"-80px", right:"-80px", width:"320px", height:"320px", borderRadius:"50%", background:"radial-gradient(circle, rgba(13,148,136,0.12) 0%, transparent 70%)", pointerEvents:"none" }} />
        <div style={{ position:"fixed", bottom:"-60px", left:"-60px", width:"260px", height:"260px", borderRadius:"50%", background:"radial-gradient(circle, rgba(13,148,136,0.08) 0%, transparent 70%)", pointerEvents:"none" }} />
        <div style={{ textAlign:"center", maxWidth:"480px", padding:"48px 32px" }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:"10px", marginBottom:"40px" }}>
            <div style={{ width:"44px", height:"44px", borderRadius:"12px", background:`linear-gradient(135deg, ${TEAL}, ${TEAL_DARK})`, display:"flex", alignItems:"center", justifyContent:"center" }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width:"22px", height:"22px" }}><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <span style={{ fontFamily:"'DM Serif Display', serif", fontSize:"22px", color:"#134e4a", letterSpacing:"-0.3px" }}>AI Caretaker</span>
          </div>
          <h1 style={{ fontFamily:"'DM Serif Display', serif", fontSize:"36px", color:"#0f172a", margin:"0 0 8px", letterSpacing:"-0.5px" }}>Welcome back</h1>
          <p style={{ color:"#64748b", fontSize:"15px", margin:"0 0 40px", lineHeight:1.6 }}>Select your profile to access the caregiver dashboard</p>
          {caregiversLoading && <p style={{ color:"#94a3b8", fontSize:"14px", marginBottom:"20px" }}>Loading caregivers…</p>}
          <div style={{ display:"flex", gap:"16px", justifyContent:"center", flexWrap:"wrap" }}>
            {caregivers.map(c => (
              <button key={c.id} className="cg-btn"
                onClick={() => { setSelectedCaregiver(c); setStep("patient"); }}
                style={{ border:"1.5px solid #e2e8f0", background:"white", borderRadius:"16px", padding:"24px 32px", cursor:"pointer", minWidth:"160px", display:"flex", flexDirection:"column", alignItems:"center", gap:"12px", boxShadow:"0 2px 8px rgba(0,0,0,0.05)" }}
              >
                <div style={{ width:"52px", height:"52px", borderRadius:"50%", background:`linear-gradient(135deg, ${TEAL_LIGHT}, #a7f3d0)`, display:"flex", alignItems:"center", justifyContent:"center", color:TEAL_DARK, fontWeight:"700", fontSize:"16px" }}>
                  {getInitials(c.first_name, c.last_name)}
                </div>
                <span style={{ color:"#1e293b", fontWeight:"600", fontSize:"15px" }}>{c.first_name} {c.last_name}</span>
                <span style={{ color:TEAL, fontSize:"12px", fontWeight:"500" }}>Log in →</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ── STEP 2: patient select ────────────────────────────────────────────────
  if (step === "patient") {
    return (
      <div style={{ fontFamily:"'DM Sans', system-ui, sans-serif", background:"#f0fdf9", minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center" }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
          * { box-sizing: border-box; }
          .pat-row { transition: all 0.15s ease; border: 1.5px solid #e2e8f0; background: white; border-radius: 12px; padding: 16px 20px; cursor: pointer; display: flex; align-items: center; justify-content: space-between; }
          .pat-row:hover { border-color: ${TEAL}; background: ${TEAL_LIGHT}; }
          .pat-row.selected { border-color: ${TEAL}; background: ${TEAL_LIGHT}; }
          .continue-btn { transition: all 0.18s ease; }
          .continue-btn:hover { opacity: 0.9; transform: translateY(-1px); }
          .continue-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        `}</style>
        <div style={{ width:"100%", maxWidth:"440px", padding:"32px 24px" }}>
          <button onClick={() => setStep("caregiver")} style={{ background:"none", border:"none", color:"#64748b", fontSize:"14px", cursor:"pointer", marginBottom:"24px", display:"flex", alignItems:"center", gap:"6px" }}>← Back</button>
          <div style={{ display:"flex", alignItems:"center", gap:"10px", marginBottom:"32px" }}>
            <div style={{ width:"36px", height:"36px", borderRadius:"10px", background:`linear-gradient(135deg, ${TEAL}, ${TEAL_DARK})`, display:"flex", alignItems:"center", justifyContent:"center" }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width:"18px", height:"18px" }}><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <span style={{ fontFamily:"'DM Serif Display', serif", fontSize:"18px", color:"#134e4a" }}>AI Caretaker</span>
          </div>
          <h2 style={{ fontFamily:"'DM Serif Display', serif", fontSize:"28px", color:"#0f172a", margin:"0 0 6px", letterSpacing:"-0.3px" }}>Select a patient</h2>
          <p style={{ color:"#64748b", fontSize:"14px", margin:"0 0 28px" }}>Hi {selectedCaregiver?.first_name} — choose who you&apos;re checking in on.</p>
          <div style={{ display:"flex", flexDirection:"column", gap:"10px", marginBottom:"28px" }}>
            {patients.length === 0 && <p style={{ color:"#94a3b8", textAlign:"center", padding:"20px 0" }}>Loading patients…</p>}
            {patients.map(p => (
              <div key={p.id} className={`pat-row${selectedPatient?.id === p.id ? " selected" : ""}`} onClick={() => setSelectedPatient(p)}>
                <div style={{ display:"flex", alignItems:"center", gap:"12px" }}>
                  <div style={{ width:"38px", height:"38px", borderRadius:"50%", background:selectedPatient?.id===p.id?`${TEAL}22`:"#f1f5f9", display:"flex", alignItems:"center", justifyContent:"center", color:TEAL_DARK, fontWeight:"600", fontSize:"13px" }}>
                    {getInitials(p.first_name, p.last_name)}
                  </div>
                  <div>
                    <div style={{ fontWeight:"600", color:"#1e293b", fontSize:"14px" }}>{p.first_name} {p.last_name}</div>
                    <div style={{ color:"#94a3b8", fontSize:"12px" }}>Patient #{p.id}</div>
                  </div>
                </div>
                <span style={{ color:TEAL }}>{Icons.chevron}</span>
              </div>
            ))}
          </div>
          <button className="continue-btn" disabled={!selectedPatient} onClick={() => setStep("dashboard")}
            style={{ width:"100%", padding:"14px", borderRadius:"12px", border:"none", background:`linear-gradient(135deg, ${TEAL}, ${TEAL_DARK})`, color:"white", fontWeight:"600", fontSize:"15px", cursor:"pointer", boxShadow:`0 4px 16px rgba(13,148,136,0.3)` }}>
            Continue to Dashboard →
          </button>
        </div>
      </div>
    );
  }

  // ── MAIN DASHBOARD ────────────────────────────────────────────────────────
  const navItems = [
    { key:"home",         label:"Home",               icon:Icons.home },
    { key:"conversation", label:"Convo History",      icon:Icons.chat },
    { key:"emergency",    label:"Emergency Log",      icon:Icons.alert },
    { key:"medicine",     label:"Medicine Log",       icon:Icons.pill },
    { key:"activity",     label:"Activity Log",       icon:Icons.activity },
    { key:"privacy",      label:"Privacy Policy",     icon:Icons.privacy },
    { key:"terms",        label:"Terms & Conditions", icon:Icons.terms },
  ];

  const patientName       = selectedPatient ? `${selectedPatient.first_name} ${selectedPatient.last_name}` : "";
  const caregiverInitials = selectedCaregiver ? getInitials(selectedCaregiver.first_name, selectedCaregiver.last_name) : "";

  const filteredLogs = logs.filter(l =>
    l.statement.toLowerCase().includes(search.toLowerCase()) ||
    l.speaker.toLowerCase().includes(search.toLowerCase())
  );

  const exportCSV = () => {
    const header = ["Date","Time","Speaker","Statement","Emergency","Severity","Activity","Medication"];
    const rows = logs.map(l => [
      fmtDate(l.timestamp), fmtTime(l.timestamp),
      l.speaker, `"${l.statement.replace(/"/g,'""')}"`,
      l.emergency_type ?? "", l.severity_score ?? "",
      l.activity_type ?? "", l.medication_type ?? "",
    ]);
    const csv = [header, ...rows].map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type:"text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href=url; a.download=`${patientName}-logs.csv`; a.click();
  };

  const pageTitle = navItems.find(n=>n.key===page)?.label ?? "";

  return (
    <div style={{ fontFamily:"'DM Sans', system-ui, sans-serif", display:"flex", height:"100vh", background:"#f8fafc", color:"#1e293b", overflow:"hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
        * { box-sizing: border-box; margin:0; padding:0; }
        ::-webkit-scrollbar { width:5px; }
        ::-webkit-scrollbar-track { background:#f1f5f9; }
        ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:10px; }
        .nav-item { display:flex; align-items:center; gap:10px; padding:9px 14px; border-radius:10px; cursor:pointer; transition:all 0.15s ease; font-size:13.5px; font-weight:500; color:#64748b; white-space:nowrap; }
        .nav-item:hover { background:#f1f5f9; color:#1e293b; }
        .nav-item.active { background:${TEAL_LIGHT}; color:${TEAL_DARK}; }
        .nav-item.active svg { stroke:${TEAL_DARK}; }
        .stat-card { background:white; border:1.5px solid #e2e8f0; border-radius:16px; padding:20px; transition:all 0.2s ease; }
        .stat-card:hover { box-shadow:0 4px 20px rgba(0,0,0,0.07); transform:translateY(-1px); }
        .log-row { border-bottom:1px solid #f1f5f9; padding:14px 0; transition:background 0.12s ease; }
        .log-row:last-child { border:none; }
        .log-row:hover { background:#fafafa; }
        .em-card { background:white; border:1.5px solid #fecaca; border-radius:14px; padding:16px 20px; margin-bottom:10px; }
        .em-card:hover { border-color:#f87171; }
        .pill-badge { display:inline-block; border-radius:100px; font-size:11px; font-weight:600; padding:3px 10px; }
        .btn-outline { display:flex; align-items:center; gap:6px; padding:9px 16px; border-radius:10px; border:1.5px solid #e2e8f0; background:white; color:#475569; font-size:13px; font-weight:500; cursor:pointer; transition:all 0.15s; }
        .btn-outline:hover { border-color:#94a3b8; background:#f8fafc; }
        .btn-teal { display:flex; align-items:center; gap:6px; padding:9px 16px; border-radius:10px; border:none; background:${TEAL}; color:white; font-size:13px; font-weight:600; cursor:pointer; transition:all 0.15s; }
        .btn-teal:hover { background:${TEAL_DARK}; }
        input[type=text]::placeholder { color:#94a3b8; }
        .fade-in { animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
      `}</style>

      {/* SIDEBAR */}
      {sidebarOpen && (
        <aside style={{ width:"220px", flexShrink:0, background:"white", borderRight:"1.5px solid #e2e8f0", display:"flex", flexDirection:"column", height:"100vh", overflow:"hidden" }}>
          <div style={{ padding:"20px 18px 16px", borderBottom:"1.5px solid #f1f5f9" }}>
            <div style={{ display:"flex", alignItems:"center", gap:"9px" }}>
              <div style={{ width:"32px", height:"32px", borderRadius:"9px", background:`linear-gradient(135deg, ${TEAL}, ${TEAL_DARK})`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width:"16px", height:"16px" }}><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
              </div>
              <div>
                <div style={{ fontFamily:"'DM Serif Display', serif", fontSize:"14px", color:"#134e4a", lineHeight:1.2 }}>AI Caretaker</div>
                <div style={{ fontSize:"10px", color:"#94a3b8", fontWeight:"500" }}>Sheets Viewer</div>
              </div>
            </div>
          </div>
          <nav style={{ flex:1, overflowY:"auto", padding:"14px 12px" }}>
            <div style={{ fontSize:"10px", fontWeight:"700", color:"#94a3b8", letterSpacing:"0.08em", textTransform:"uppercase", padding:"0 2px 8px" }}>Navigation</div>
            {navItems.map(n => (
              <div key={n.key} className={`nav-item${page===n.key?" active":""}`} onClick={() => setPage(n.key)}>
                {n.icon}{n.label}
                {n.key==="home" && page==="home" && <div style={{ width:"6px", height:"6px", borderRadius:"50%", background:TEAL, marginLeft:"auto" }} />}
              </div>
            ))}
            <div style={{ marginTop:"20px", background:"#f8fafc", border:"1.5px solid #e2e8f0", borderRadius:"12px", padding:"12px 14px" }}>
              <div style={{ fontSize:"12px", fontWeight:"700", color:"#475569", marginBottom:"4px" }}>Read-only</div>
              <div style={{ fontSize:"11.5px", color:"#94a3b8", lineHeight:1.5 }}>Data is pulled from Supabase through the backend API.</div>
            </div>
          </nav>
          <div style={{ padding:"14px 12px", borderTop:"1.5px solid #f1f5f9", display:"flex", alignItems:"center", gap:"10px" }}>
            <div style={{ width:"32px", height:"32px", borderRadius:"50%", background:TEAL_LIGHT, display:"flex", alignItems:"center", justifyContent:"center", color:TEAL_DARK, fontWeight:"700", fontSize:"12px", flexShrink:0 }}>
              {caregiverInitials}
            </div>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ fontSize:"12.5px", fontWeight:"600", color:"#1e293b", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{selectedCaregiver?.first_name}…</div>
              <div style={{ fontSize:"11px", color:"#94a3b8" }}>Caregiver</div>
            </div>
            <button onClick={() => { setStep("caregiver"); setSelectedCaregiver(null); setSelectedPatient(null); setLogs([]); }} title="Log out" style={{ background:"none", border:"none", cursor:"pointer", color:"#94a3b8", padding:"4px" }}>
              {Icons.logout}
            </button>
          </div>
        </aside>
      )}

      {/* MAIN CONTENT */}
      <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>
        <header style={{ height:"56px", borderBottom:"1.5px solid #e2e8f0", background:"white", display:"flex", alignItems:"center", padding:"0 24px", gap:"14px", flexShrink:0 }}>
          <button onClick={() => setSidebarOpen(v=>!v)} style={{ background:"none", border:"none", cursor:"pointer", color:"#64748b", display:"flex", lineHeight:1 }}>{Icons.menu}</button>
          <div style={{ flex:1 }}>
            <div style={{ fontWeight:"700", fontSize:"14px", color:"#1e293b" }}>{pageTitle}</div>
            {page==="home"         && <div style={{ fontSize:"11.5px", color:"#94a3b8" }}>Monitoring cognitive wellness and safety through real-time conversational AI.</div>}
            {page==="conversation" && <div style={{ fontSize:"11.5px", color:"#94a3b8" }}>Conversation entries captured over time</div>}
            {page==="emergency"    && <div style={{ fontSize:"11.5px", color:"#94a3b8" }}>High-severity events flagged by AI</div>}
            {page==="medicine"     && <div style={{ fontSize:"11.5px", color:"#94a3b8" }}>Medication mentions detected in conversations</div>}
            {page==="activity"     && <div style={{ fontSize:"11.5px", color:"#94a3b8" }}>Activities detected from patient speech</div>}
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:"8px", background:"#f1f5f9", borderRadius:"100px", padding:"6px 14px 6px 8px" }}>
            <div style={{ width:"26px", height:"26px", borderRadius:"50%", background:TEAL_LIGHT, display:"flex", alignItems:"center", justifyContent:"center", color:TEAL_DARK, fontWeight:"700", fontSize:"11px" }}>
              {selectedPatient ? getInitials(selectedPatient.first_name, selectedPatient.last_name) : ""}
            </div>
            <span style={{ fontSize:"13px", fontWeight:"600", color:"#1e293b" }}>{patientName}</span>
          </div>
          <button className="btn-outline" onClick={refreshLogs}>{Icons.refresh} Refresh</button>
        </header>

        <main style={{ flex:1, overflowY:"auto", padding:"28px 32px" }} className="fade-in">

          {/* HOME */}
          {page==="home" && (
            <div>
              <div style={{ background:`linear-gradient(135deg, #f0fdf9 0%, #ccfbf1 100%)`, border:`1.5px solid #99f6e4`, borderRadius:"20px", padding:"32px 36px", marginBottom:"28px", position:"relative", overflow:"hidden" }}>
                <div style={{ position:"absolute", top:"-40px", right:"-40px", width:"180px", height:"180px", borderRadius:"50%", background:"rgba(13,148,136,0.07)", pointerEvents:"none" }} />
                <h1 style={{ fontFamily:"'DM Serif Display', serif", fontSize:"32px", color:"#0f172a", letterSpacing:"-0.5px", marginBottom:"6px" }}>AI Caretaker <span style={{ color:TEAL }}>Data Log</span></h1>
                <p style={{ color:"#64748b", fontSize:"14.5px", marginBottom:"24px", lineHeight:1.6 }}>Monitoring cognitive wellness and safety through real-time conversational AI.</p>
                <div style={{ display:"flex", gap:"12px" }}>
                  <button className="btn-teal" onClick={() => setPage("conversation")} style={{ borderRadius:"10px", padding:"10px 20px", fontSize:"14px" }}>Open Convo History →</button>
                  <button className="btn-outline" onClick={() => setPage("emergency")} style={{ borderRadius:"10px", padding:"10px 20px", fontSize:"14px" }}>Emergency Log →</button>
                </div>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"16px", marginBottom:"28px" }}>
                {[
                  { label:"CONVERSATIONS", today: logs.filter(l=>{try{return new Date(l.timestamp).toDateString()===new Date().toDateString()}catch{return false}}).length, total:logs.length, color:"#0d9488", bg:TEAL_LIGHT, icon:Icons.chat },
                  { label:"EMERGENCIES",   today: emergencies.filter(l=>{try{return new Date(l.timestamp).toDateString()===new Date().toDateString()}catch{return false}}).length, total:emergencies.length, color:"#dc2626", bg:"#fee2e2", icon:Icons.alert },
                  { label:"MEDICINES",     today: medicines.filter(l=>{try{return new Date(l.timestamp).toDateString()===new Date().toDateString()}catch{return false}}).length, total:medicines.length, color:"#059669", bg:"#d1fae5", icon:Icons.pill },
                  { label:"ACTIVITIES",   today: activities.filter(l=>{try{return new Date(l.timestamp).toDateString()===new Date().toDateString()}catch{return false}}).length, total:activities.length, color:"#7c3aed", bg:"#ede9fe", icon:Icons.activity },
                ].map(s => (
                  <div key={s.label} className="stat-card" style={{ display:"flex", alignItems:"flex-start", gap:"16px" }}>
                    <div style={{ width:"40px", height:"40px", borderRadius:"12px", background:s.bg, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                      <span style={{ color:s.color }}>{s.icon}</span>
                    </div>
                    <div>
                      <div style={{ fontSize:"10.5px", fontWeight:"700", color:"#94a3b8", letterSpacing:"0.06em", marginBottom:"4px" }}>{s.label}</div>
                      <div style={{ display:"flex", alignItems:"baseline", gap:"6px" }}>
                        <span style={{ fontSize:"26px", fontWeight:"700", color:"#1e293b", lineHeight:1 }}>{s.today}</span>
                        <span style={{ fontSize:"12px", color:"#94a3b8" }}>today</span>
                        <span style={{ fontSize:"12px", color:"#94a3b8" }}>/</span>
                        <span style={{ fontSize:"14px", fontWeight:"600", color:"#475569" }}>{s.total}</span>
                        <span style={{ fontSize:"12px", color:"#94a3b8" }}>total</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 380px", gap:"20px" }}>
                <div style={{ background:"white", border:"1.5px solid #e2e8f0", borderRadius:"16px", padding:"20px 24px" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"18px" }}>
                    <span style={{ color:TEAL }}>{Icons.chat}</span>
                    <span style={{ fontWeight:"700", fontSize:"14px" }}>Recent Conversations</span>
                  </div>
                  {loading && <p style={{ color:"#94a3b8", textAlign:"center", padding:"20px 0", fontSize:"13px" }}>Loading…</p>}
                  {logs.slice(0, 8).map((l, i) => (
                    <div key={i} className="log-row" style={{ display:"flex", gap:"12px" }}>
                      <div style={{ width:"32px", height:"32px", borderRadius:"50%", background:l.speaker==="User"?"#f1f5f9":TEAL_LIGHT, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0, fontSize:"11px", fontWeight:"700", color:l.speaker==="User"?"#64748b":TEAL_DARK }}>
                        {l.speaker==="User"?"P":"A"}
                      </div>
                      <div style={{ flex:1, minWidth:0 }}>
                        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"2px" }}>
                          <span style={{ fontWeight:"600", fontSize:"13px" }}>{l.speaker==="User"?"Patient":"Assistant"}</span>
                          <span style={{ fontSize:"11px", color:"#94a3b8" }}>{fmtTime(l.timestamp)}</span>
                        </div>
                        <p style={{ fontSize:"13px", color:"#475569", lineHeight:1.5, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{l.statement}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ background:"white", border:"1.5px solid #e2e8f0", borderRadius:"16px", padding:"20px 24px" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"18px" }}>
                    <span style={{ color:"#dc2626" }}>{Icons.alert}</span>
                    <span style={{ fontWeight:"700", fontSize:"14px" }}>Recent Emergencies</span>
                  </div>
                  {emergencies.length === 0 && (
                    <div style={{ textAlign:"center", padding:"30px 0", color:"#94a3b8", fontSize:"13px" }}>
                      <div style={{ fontSize:"28px", marginBottom:"8px" }}>✅</div>No emergencies recorded
                    </div>
                  )}
                  {emergencies.slice(0, 5).map((e, i) => (
                    <div key={i} style={{ marginBottom:"10px", background:"#fff5f5", border:"1.5px solid #fecaca", borderRadius:"12px", padding:"12px 14px" }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"4px" }}>
                        <span className="pill-badge" style={{ background:"#fee2e2", color:"#dc2626" }}>{e.emergency_type}</span>
                        <span style={{ fontSize:"11px", color:"#94a3b8" }}>{fmtDate(e.timestamp)}</span>
                      </div>
                      <p style={{ fontSize:"12.5px", color:"#7f1d1d", lineHeight:1.5, marginTop:"4px" }}>{e.statement}</p>
                      {e.severity_score !== undefined && (
                        <div style={{ marginTop:"8px", display:"flex", alignItems:"center", gap:"8px" }}>
                          <div style={{ flex:1, height:"4px", borderRadius:"4px", background:"#fee2e2", overflow:"hidden" }}>
                            <div style={{ width:`${e.severity_score}%`, height:"100%", background:"#ef4444", borderRadius:"4px" }} />
                          </div>
                          <span style={{ fontSize:"11px", fontWeight:"700", color:"#dc2626" }}>{e.severity_score}/100</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* CONVERSATION LOG */}
          {page==="conversation" && (
            <div>
              <div style={{ display:"flex", alignItems:"center", gap:"12px", marginBottom:"20px" }}>
                <div style={{ flex:1, display:"flex", alignItems:"center", gap:"10px", background:"white", border:"1.5px solid #e2e8f0", borderRadius:"12px", padding:"10px 14px" }}>
                  <span style={{ color:"#94a3b8" }}>{Icons.search}</span>
                  <input type="text" placeholder="Search conversations…" value={search} onChange={e => setSearch(e.target.value)} style={{ border:"none", outline:"none", flex:1, fontSize:"13.5px", color:"#1e293b", background:"transparent" }} />
                </div>
                <div style={{ background:"#f1f5f9", borderRadius:"10px", padding:"6px 14px", fontSize:"13px", fontWeight:"600", color:"#64748b" }}>{filteredLogs.length} rows</div>
                <button className="btn-outline" onClick={exportCSV}>{Icons.download} Export CSV</button>
              </div>
              <div style={{ background:"white", border:"1.5px solid #e2e8f0", borderRadius:"16px", overflow:"hidden" }}>
                <div style={{ padding:"14px 24px", borderBottom:"1.5px solid #f1f5f9" }}>
                  <span style={{ fontWeight:"700", fontSize:"15px" }}>Convo History</span>
                  <span style={{ color:"#94a3b8", fontSize:"12.5px", marginLeft:"10px" }}>4 cols · {filteredLogs.length} rows</span>
                </div>
                <div style={{ overflowX:"auto" }}>
                  <table style={{ width:"100%", borderCollapse:"collapse" }}>
                    <thead>
                      <tr style={{ background:"#f8fafc" }}>
                        {["DATE","TIME","SENDER","STATEMENT"].map(h => (
                          <th key={h} style={{ padding:"10px 20px", textAlign:"left", fontSize:"11px", fontWeight:"700", color:"#94a3b8", letterSpacing:"0.06em", borderBottom:"1.5px solid #f1f5f9", whiteSpace:"nowrap" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {filteredLogs.map((l, i) => (
                        <tr key={i} style={{ borderBottom:"1px solid #f8fafc" }}>
                          <td style={{ padding:"13px 20px", fontSize:"13px", color:"#475569", whiteSpace:"nowrap" }}>{fmtDate(l.timestamp)}</td>
                          <td style={{ padding:"13px 20px", fontSize:"13px", color:"#475569", whiteSpace:"nowrap" }}>{fmtTime(l.timestamp)}</td>
                          <td style={{ padding:"13px 20px", whiteSpace:"nowrap" }}>
                            <span className="pill-badge" style={{ background:l.speaker==="User"?"#f1f5f9":TEAL_LIGHT, color:l.speaker==="User"?"#475569":TEAL_DARK }}>{l.speaker==="User"?"Patient":l.speaker}</span>
                          </td>
                          <td style={{ padding:"13px 20px", fontSize:"13px", color:"#1e293b", lineHeight:1.55, maxWidth:"520px" }}>{l.statement}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* EMERGENCY LOG */}
          {page==="emergency" && (
            <div>
              {emergencies.length === 0 && (
                <div style={{ textAlign:"center", padding:"60px 0", color:"#94a3b8" }}>
                  <div style={{ fontSize:"48px", marginBottom:"12px" }}>✅</div>
                  <div style={{ fontWeight:"700", fontSize:"16px", color:"#475569", marginBottom:"6px" }}>No emergencies</div>
                  <div style={{ fontSize:"13px" }}>No emergency events have been detected for this patient.</div>
                </div>
              )}
              <div style={{ display:"grid", gap:"12px" }}>
                {emergencies.map((e, i) => (
                  <div key={i} className="em-card">
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:"10px" }}>
                      <div style={{ display:"flex", alignItems:"center", gap:"10px" }}>
                        <div style={{ width:"36px", height:"36px", borderRadius:"10px", background:"#fee2e2", display:"flex", alignItems:"center", justifyContent:"center", color:"#dc2626" }}>{Icons.alert}</div>
                        <div>
                          <span className="pill-badge" style={{ background:"#fee2e2", color:"#dc2626", fontSize:"12px" }}>{e.emergency_type}</span>
                          <div style={{ fontSize:"11px", color:"#94a3b8", marginTop:"2px" }}>{fmtDate(e.timestamp)} at {fmtTime(e.timestamp)}</div>
                        </div>
                      </div>
                      {e.severity_score !== undefined && (
                        <div style={{ textAlign:"right" }}>
                          <div style={{ fontSize:"11px", color:"#94a3b8", marginBottom:"4px" }}>Severity</div>
                          <div style={{ fontWeight:"800", fontSize:"22px", color: e.severity_score>=70?"#dc2626":e.severity_score>=40?"#f59e0b":"#22c55e", lineHeight:1 }}>{e.severity_score}<span style={{ fontSize:"13px", fontWeight:"500" }}>/100</span></div>
                        </div>
                      )}
                    </div>
                    <p style={{ fontSize:"13.5px", color:"#475569", lineHeight:1.6, padding:"10px 14px", background:"#fff5f5", borderRadius:"10px" }}>{e.statement}</p>
                    {e.severity_score !== undefined && (
                      <div style={{ marginTop:"10px", height:"5px", borderRadius:"5px", background:"#fee2e2", overflow:"hidden" }}>
                        <div style={{ width:`${e.severity_score}%`, height:"100%", background: e.severity_score>=70?"#ef4444":e.severity_score>=40?"#f59e0b":"#22c55e", borderRadius:"5px", transition:"width 0.6s ease" }} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* MEDICINE LOG */}
          {page==="medicine" && (
            <div>
              {medicines.length === 0 && (
                <div style={{ textAlign:"center", padding:"60px 0", color:"#94a3b8" }}>
                  <div style={{ fontSize:"48px", marginBottom:"12px" }}>💊</div>
                  <div style={{ fontWeight:"700", fontSize:"16px", color:"#475569", marginBottom:"6px" }}>No medication events</div>
                  <div style={{ fontSize:"13px" }}>No medication mentions have been detected yet.</div>
                </div>
              )}
              <div style={{ display:"grid", gap:"12px" }}>
                {medicines.map((m, i) => (
                  <div key={i} style={{ background:"white", border:"1.5px solid #d1fae5", borderRadius:"14px", padding:"16px 20px" }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"8px" }}>
                      <span className="pill-badge" style={{ background:"#d1fae5", color:"#059669" }}>{m.medication_type}</span>
                      <span style={{ fontSize:"11px", color:"#94a3b8" }}>{fmtDate(m.timestamp)} {fmtTime(m.timestamp)}</span>
                    </div>
                    <p style={{ fontSize:"13.5px", color:"#475569", lineHeight:1.6 }}>{m.statement}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ACTIVITY LOG */}
          {page==="activity" && (
            <div>
              {activities.length === 0 && (
                <div style={{ textAlign:"center", padding:"60px 0", color:"#94a3b8" }}>
                  <div style={{ fontSize:"48px", marginBottom:"12px" }}>🏃</div>
                  <div style={{ fontWeight:"700", fontSize:"16px", color:"#475569", marginBottom:"6px" }}>No activities logged</div>
                  <div style={{ fontSize:"13px" }}>Activity mentions will appear here.</div>
                </div>
              )}
              <div style={{ display:"grid", gap:"12px" }}>
                {activities.map((a, i) => (
                  <div key={i} style={{ background:"white", border:"1.5px solid #ede9fe", borderRadius:"14px", padding:"16px 20px" }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"8px" }}>
                      <span className="pill-badge" style={{ background:"#ede9fe", color:"#7c3aed" }}>{a.activity_type}</span>
                      <span style={{ fontSize:"11px", color:"#94a3b8" }}>{fmtDate(a.timestamp)} {fmtTime(a.timestamp)}</span>
                    </div>
                    <p style={{ fontSize:"13.5px", color:"#475569", lineHeight:1.6 }}>{a.statement}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* PRIVACY / TERMS */}
          {(page==="privacy"||page==="terms") && (
            <div style={{ maxWidth:"640px" }}>
              <div style={{ background:"white", border:"1.5px solid #e2e8f0", borderRadius:"16px", padding:"32px 36px" }}>
                <h2 style={{ fontFamily:"'DM Serif Display', serif", fontSize:"24px", color:"#0f172a", marginBottom:"16px" }}>
                  {page==="privacy"?"Privacy Policy":"Terms & Conditions"}
                </h2>
                <p style={{ color:"#64748b", fontSize:"14px", lineHeight:1.8 }}>
                  {page==="privacy"
                    ? "This dashboard displays patient conversation data for authorized caregivers only. All data is stored securely in Supabase and accessed through encrypted API endpoints. No data is shared with third parties. Access is restricted to registered caregivers."
                    : "By using the AI Caretaker dashboard, you agree to use this tool solely for legitimate caregiving purposes. Unauthorized access or sharing of patient data is prohibited. This tool is intended as a supplementary monitoring aid and does not replace professional medical judgment."}
                </p>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
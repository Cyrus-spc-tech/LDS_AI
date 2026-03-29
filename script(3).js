/**
 * LexiBrief Pro - Enterprise Legal AI
 * Frontend Core Logic
 */

// --- 1. State Management ---
let currentUser = "John Doe";
let history = [
    { 
        name: "Service_Agreement_v1.pdf", 
        date: "Mar 25, 2026", 
        risk: "Low", 
        summ: "Standard SLA with 99.9% uptime guarantee and Net-30 payment terms.",
        juris: "New York, USA",
        term: "30 Days Notice"
    },
    { 
        name: "Compliance_Audit_Internal.docx", 
        date: "Mar 20, 2026", 
        risk: "High", 
        summ: "Audit flags missing data encryption protocols in Section 4.2 of the security policy.",
        juris: "California, USA",
        term: "Immediate on Breach"
    }
];

// --- 2. Element Selectors ---
const appShell = document.getElementById('app-shell');
const authPage = document.getElementById('page-auth');
const reportModal = document.getElementById('report-modal');

// --- 3. Authentication Logic ---
window.toggleAuth = function(type) {
    document.getElementById('form-login').classList.toggle('hidden', type === 'register');
    document.getElementById('form-register').classList.toggle('hidden', type === 'login');
};

window.handleAuth = function(e, type) {
    e.preventDefault();
    const nameInput = document.getElementById('reg-name');
    currentUser = (type === 'register' && nameInput.value) ? nameInput.value : "John Doe";
    
    // Transition to Dashboard
    authPage.classList.add('hidden');
    appShell.classList.remove('hidden');
    document.getElementById('user-display-name').innerText = `${currentUser}'s Profile`;
    
    // Initialize Icons
    lucide.createIcons();
};

// --- 4. Navigation Logic ---
window.showPage = function(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => page.classList.add('hidden'));
    // Show selected page
    document.getElementById('page-' + pageId).classList.remove('hidden');
    
    // Update Sidebar Active State
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    event.currentTarget.classList.add('active');
    
    if(pageId === 'profile') renderHistory();
};

// --- 5. File Processing Logic ---
window.processFile = function(input) {
    if(!input.files) return;
    const fileName = input.files.name;
    
    // 1. Simulate Processing Feedback
    console.log(`Neural Engine analyzing: ${fileName}`);
    
    // 2. Simulate AI Delay (2 seconds)
    setTimeout(() => {
        const newSummary = `AI Synthesis for ${fileName}: This document has been verified for structural integrity. Key obligations include confidentiality for 24 months post-termination.`;
        
        const newEntry = {
            name: fileName,
            date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
            risk: "Medium",
            summ: newSummary,
            juris: "Standard Federal",
            term: "60 Days"
        };

        // Update Data
        history.unshift(newEntry);
        
        // Auto-Open the Summary Section
        openSummary(newEntry.name, newEntry.summ, "Medium Risk: Section 5.1 lacks specific indemnification limits.", newEntry.juris, newEntry.term);
        
        // Update stats if on profile
        renderHistory();
    }, 2000);
};

// --- 6. Summary Report Logic ---
window.openSummary = function(name, summary, risks, juris, term) {
    // Find item in history if only name is passed (for history clicks)
    const item = history.find(h => h.name === name);
    
    document.getElementById('m-name').innerText = name;
    document.getElementById('m-summary').innerText = summary || item.summ;
    document.getElementById('m-risk').innerHTML = `<b>Risk Alert:</b> ${risks || "Section 4.2 contains potential liability gaps."}`;
    
    // Show Modal
    reportModal.classList.remove('hidden');
    lucide.createIcons();
};

window.closeModal = function() {
    reportModal.classList.add('hidden');
};

// --- 7. Data Rendering ---
function renderHistory() {
    const body = document.getElementById('history-body');
    if(!body) return;

    body.innerHTML = history.map(item => `
        <tr onclick="openSummary('${item.name}', '${item.summ}')" style="cursor:pointer">
            <td><div style="display:flex; align-items:center; gap:10px;"><i data-lucide="file-text" size="16"></i> <b>${item.name}</b></div></td>
            <td>${item.date}</td>
            <td><span class="risk-tag ${item.risk === 'Low' ? 'risk-low' : 'risk-high'}">${item.risk}</span></td>
            <td style="color: var(--primary); font-weight:600;">View Report</td>
        </tr>
    `).join('');
    
    lucide.createIcons();
}

// Initializing UI components on load
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
});
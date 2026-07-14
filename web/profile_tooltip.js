/**
 * Star Model Converter Pro - Profile Tooltip Extension
 * Displays profile metadata as tooltips when hovering over profile dropdown
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const profileCache = {};
let currentTooltip = null;
let hoverTimeout = null;
let lastHoveredValue = null;

async function loadProfileMetadata(profileName) {
    if (profileCache[profileName]) return profileCache[profileName];
    
    try {
        const response = await api.fetchApi(`/starnodes/profile/${encodeURIComponent(profileName)}`);
        if (response.ok) {
            const data = await response.json();
            profileCache[profileName] = data.__metadata__ || {};
            return profileCache[profileName];
        }
    } catch (error) {
        console.error(`Failed to load profile ${profileName}:`, error);
    }
    
    return null;
}

// Hilfsfunktion zum sauberen Entfernen des Tooltips
function removeTooltip() {
    if (currentTooltip && currentTooltip.parentNode) {
        currentTooltip.parentNode.removeChild(currentTooltip);
    }
    currentTooltip = null;
}

function createTooltip(metadata, event) {
    removeTooltip(); // Alten Tooltip sicherheitshalber löschen
    
    const tooltip = document.createElement("div");
    tooltip.className = "starnodes-profile-tooltip";
    
    // WICHTIG: position: fixed macht den Tooltip immun gegen Canvas-Zoom/Pan
    tooltip.style.cssText = `
        position: fixed;
        background: rgba(0, 0, 0, 0.9);
        color: #fff;
        padding: 12px;
        border-radius: 6px;
        font-size: 12px;
        line-height: 1.6;
        z-index: 10000;
        pointer-events: none;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    `;
    
    const lines = [
        `<div style="font-weight: bold; margin-bottom: 8px; color: #4CAF50;">📋 Profile Information</div>`,
        `<div><strong>Model:</strong> ${metadata.original_model_name || 'Unknown'}</div>`,
        `<div><strong>Created:</strong> ${formatDate(metadata.timestamp)}</div>`,
        `<div><strong>Layers:</strong> ${metadata.total_layers || 'Unknown'}</div>`,
        `<div style="margin-top: 8px; font-size: 10px; color: #888;">Created by: ${metadata.created_by || 'Unknown'}</div>`
    ];
    
    tooltip.innerHTML = lines.join('');
    document.body.appendChild(tooltip);
    
    // Position direkt beim Erstellen setzen
    positionTooltip(tooltip, event);
    
    return tooltip;
}

function positionTooltip(tooltip, event) {
    if (!tooltip || !event) return;
    
    const offset = 15; // Abstand zum Mauszeiger
    let x = event.clientX + offset;
    let y = event.clientY + offset;
    
    // Verhindern, dass der Tooltip aus dem Bildschirm ragt
    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - offset;
    if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - offset;
    
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
}

function formatDate(isoString) {
    if (!isoString) return 'Unknown';
    try {
        return new Date(isoString).toLocaleString();
    } catch {
        return isoString;
    }
}

app.registerExtension({
    name: "StarNodes.ProfileTooltip",
    
    async nodeCreated(node) {
        if (node.comfyClass === "StarUltimateModelConverterPro") {
            const profileWidget = node.widgets?.find(w => w.name === "profile");
            
            if (profileWidget) {
                const originalMouse = profileWidget.mouse;
                
                // 1. Tooltip, wenn man über das geschlossene Widget in der Node fährt
                profileWidget.mouse = function(event, pos, node) {
                    if (originalMouse) originalMouse.apply(this, arguments);
                    
                    if (event.type === "pointermove") {
                        const currentValue = profileWidget.value;
                        
                        // Tooltip an der Maus kleben lassen
                        if (currentTooltip) {
                            positionTooltip(currentTooltip, event);
                        }
                        
                        // Nur neu laden, wenn sich der Wert geändert hat
                        if (currentValue && currentValue !== "No profiles found" && currentValue !== lastHoveredValue) {
                            lastHoveredValue = currentValue;
                            if (hoverTimeout) clearTimeout(hoverTimeout);
                            
                            hoverTimeout = setTimeout(async () => {
                                const metadata = await loadProfileMetadata(currentValue);
                                if (metadata && lastHoveredValue === currentValue) {
                                    currentTooltip = createTooltip(metadata, event);
                                }
                            }, 300);
                        }
                    }
                };
                
                // Aufräumen, wenn die Maus die Node komplett verlässt
                const originalOnMouseLeave = node.onMouseLeave;
                node.onMouseLeave = function(event) {
                    if (originalOnMouseLeave) originalOnMouseLeave.apply(this, arguments);
                    if (hoverTimeout) clearTimeout(hoverTimeout);
                    removeTooltip();
                    lastHoveredValue = null;
                };
            }
        }
    }
});

// 2. Globale Listener für das geöffnete Dropdown-Menü (LiteGraph ContextMenu)
document.addEventListener("mouseover", (e) => {
    const target = e.target;
    
    // LiteGraph Dropdown-Items haben immer die Klasse "litemenu-entry"
    if (target && target.classList && target.classList.contains("litemenu-entry")) {
        const profileName = target.textContent;
        
        // Prüfen, ob der Eintrag ein Profil ist (endet auf .json)
        if (profileName.endsWith(".json")) {
            if (hoverTimeout) clearTimeout(hoverTimeout);
            
            hoverTimeout = setTimeout(async () => {
                const metadata = await loadProfileMetadata(profileName);
                if (metadata) {
                    currentTooltip = createTooltip(metadata, e);
                }
            }, 200); // Etwas schnelleres Feedback im Menü
        }
    }
});

document.addEventListener("mousemove", (e) => {
    // Wenn das Menü offen ist, positioniere den Tooltip bei Bewegung nach
    if (currentTooltip && e.target && e.target.classList && e.target.classList.contains("litemenu-entry")) {
        positionTooltip(currentTooltip, e);
    }
});

document.addEventListener("mouseout", (e) => {
    const target = e.target;
    // Wenn wir ein Dropdown-Item verlassen, Tooltip sofort ausblenden
    if (target && target.classList && target.classList.contains("litemenu-entry")) {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        removeTooltip();
    }
});

// Globale Styles hinzufügen
const style = document.createElement("style");
style.textContent = `
    .starnodes-profile-tooltip {
        animation: fadeIn 0.1s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
`;
document.head.appendChild(style);
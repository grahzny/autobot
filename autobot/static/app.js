/* Autobot -- Living Entity Frontend */

let personId = null;
let personName = null;
let ws = null;
let debugInterval = null;
let thinkingHistory = [];

// ======================================================================
// Connect
// ======================================================================

document.getElementById('connect-btn').addEventListener('click', doConnect);
document.getElementById('name-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') doConnect();
});

async function doConnect() {
  const name = document.getElementById('name-input').value.trim();
  if (!name) return;
  personName = name;

  try {
    const res = await fetch('/api/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ person_name: name }),
    });
    const data = await res.json();
    personId = data.person_id;

    document.getElementById('entity-name').textContent = data.entity_name;
    document.getElementById('entity-mood-hint').textContent = data.mood_hint;

    document.getElementById('connect-screen').classList.add('hidden');
    document.getElementById('chat-screen').classList.remove('hidden');
    document.getElementById('msg-input').focus();

    connectWebSocket();
    startDebugPolling();
  } catch (err) {
    console.error('Connect failed:', err);
  }
}

// ======================================================================
// WebSocket
// ======================================================================

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/${personId}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'connected') {
      document.getElementById('entity-mood-hint').textContent = data.mood_hint;
    } else if (data.type === 'message') {
      addMessage('entity', data.text, data.mood_hint, data.thinking);
    } else if (data.type === 'thinking_update') {
      addThought(data.thinking, data.decision, data.tick);
      // Update mood/energy from thinking update
      if (data.mood) {
        document.getElementById('mood-val').textContent = data.mood;
        document.getElementById('mood-val').className = 'mood-badge mood-' + data.mood;
        document.getElementById('entity-mood-hint').textContent = data.mood;
      }
      if (data.energy !== undefined) {
        updateEnergyDisplay(data.energy);
      }
    } else if (data.type === 'received') {
      // Our message was received
    }
  };

  ws.onclose = () => {
    console.log('WebSocket closed, reconnecting in 3s...');
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };
}

// ======================================================================
// Sending messages
// ======================================================================

document.getElementById('send-btn').addEventListener('click', doSend);
document.getElementById('msg-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSend();
});

function doSend() {
  const input = document.getElementById('msg-input');
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

  ws.send(JSON.stringify({ type: 'message', text }));
  addMessage('human', text);
  input.value = '';
}

// ======================================================================
// Chat display
// ======================================================================

function addMessage(role, text, moodHint, thinking) {
  const container = document.getElementById('messages');
  const bubble = document.createElement('div');
  bubble.className = `msg msg-${role}`;

  const content = document.createElement('div');
  content.className = 'msg-content';
  content.textContent = text;
  bubble.appendChild(content);

  if (role === 'entity' && moodHint) {
    const mood = document.createElement('span');
    mood.className = 'msg-mood';
    mood.textContent = moodHint;
    bubble.appendChild(mood);
  }

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;

  // Feed entity thinking into the thinking log
  if (role === 'entity' && thinking) {
    addThought(thinking, 'respond', null);
  }
}

// ======================================================================
// Thinking history
// ======================================================================

function addThought(text, decision, tick) {
  // Deduplicate: skip if the last entry has the same text
  if (thinkingHistory.length > 0 && thinkingHistory[thinkingHistory.length - 1].text === text) {
    return;
  }

  thinkingHistory.push({ text, decision, tick, time: new Date() });
  // Keep last 50
  if (thinkingHistory.length > 50) thinkingHistory = thinkingHistory.slice(-50);
  renderThinkingHistory();
}

function renderThinkingHistory() {
  const box = document.getElementById('thinking-box');
  if (thinkingHistory.length === 0) {
    box.textContent = '--';
    return;
  }

  box.innerHTML = '';
  for (const t of thinkingHistory) {
    const entry = document.createElement('div');
    entry.className = 'thought-entry thought-' + t.decision;
    const timeStr = t.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const label = t.decision === 'reflect' ? 'reflect'
                : t.decision === 'respond' ? 'response'
                : t.decision === 'proactive' ? 'proactive'
                : t.decision === 'idle' ? 'idle'
                : 'thought';
    entry.innerHTML = `<span class="thought-label">[${timeStr} ${label}]</span> ${escapeHtml(t.text)}`;
    box.appendChild(entry);
  }
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ======================================================================
// Energy display helper
// ======================================================================

function updateEnergyDisplay(energy) {
  const energyPct = Math.round(energy * 100);
  const energyBar = document.getElementById('energy-bar');
  energyBar.style.width = energyPct + '%';
  energyBar.className = 'bar' + (energyPct < 30 ? ' bar-low' : energyPct < 60 ? ' bar-mid' : '');
  document.getElementById('energy-pct').textContent = energyPct + '%';
}

// ======================================================================
// Debug panel polling
// ======================================================================

document.getElementById('toggle-debug').addEventListener('click', () => {
  const panel = document.getElementById('debug-panel');
  const btn = document.getElementById('toggle-debug');
  panel.classList.toggle('hidden');
  btn.textContent = panel.classList.contains('hidden') ? 'Show Debug' : 'Hide Debug';
});

function startDebugPolling() {
  updateDebugPanel();
  debugInterval = setInterval(updateDebugPanel, 5000);
}

async function updateDebugPanel() {
  try {
    const res = await fetch('/api/debug/state');
    const data = await res.json();

    // Update mood
    document.getElementById('mood-val').textContent = data.dominant_mood;
    document.getElementById('mood-val').className = 'mood-badge mood-' + data.dominant_mood;
    document.getElementById('entity-mood-hint').textContent = data.dominant_mood;

    // Update energy
    updateEnergyDisplay(data.energy);

    // Emotions detail
    const emotionsDiv = document.getElementById('emotions-detail');
    if (data.emotions) {
      const emotionColors = {
        anxiety: '#fbbf24',
        optimism: '#22c55e',
        irritability: '#f87171',
        social_warmth: '#38bdf8',
        avoidance_bias: '#818cf8',
        risk_tolerance: '#a78bfa',
      };
      emotionsDiv.innerHTML = Object.entries(data.emotions).map(([key, val]) => {
        const pct = Math.round(val * 100);
        const color = emotionColors[key] || '#64748b';
        const label = key.replace(/_/g, ' ');
        return `<div class="emotion-item">
          <span class="emotion-name">${label}</span>
          <div class="emotion-bar-wrap">
            <div class="emotion-bar" style="width:${pct}%;background:${color}"></div>
          </div>
          <span class="emotion-val">${pct}%</span>
        </div>`;
      }).join('');
    }

    // Goals
    const goalsList = document.getElementById('goals-list');
    const activeGoals = (data.goals || []).filter(g => !g.completed && !g.abandoned);
    if (activeGoals.length) {
      goalsList.innerHTML = activeGoals.map(g => {
        const pPct = Math.round((g.priority || 0) * 100);
        return `<li class="goal-item">
          <div class="goal-header">
            <span><span class="goal-cat">[${g.category}]</span> ${escapeHtml(g.description)}</span>
          </div>
          <div class="goal-priority-bar">
            <div class="goal-priority-fill" style="width:${pPct}%"></div>
          </div>
        </li>`;
      }).join('');
    } else {
      goalsList.innerHTML = '<li class="dim">No active goals</li>';
    }

    // Memories
    const memList = document.getElementById('memories-list');
    if (data.memories && data.memories.length) {
      memList.innerHTML = data.memories.map(m => {
        const valenceClass = m.valence > 0.1 ? 'positive' : m.valence < -0.1 ? 'negative' : 'neutral';
        const valenceLabel = m.valence > 0.1 ? '+' : m.valence < -0.1 ? '-' : '~';
        const arousalLabel = m.arousal > 0.5 ? 'vivid' : 'faint';
        const unresolvedTag = m.unresolved ? '<span class="memory-tag negative">unresolved</span>' : '';
        return `<li class="memory-item">
          <div class="memory-summary">${escapeHtml(m.summary)}</div>
          <div class="memory-meta">
            <span class="memory-tag ${valenceClass}" title="valence: ${m.valence.toFixed(2)}">${valenceLabel} ${arousalLabel}</span>
            ${unresolvedTag}
          </div>
        </li>`;
      }).join('');
    } else {
      memList.innerHTML = '<li class="dim">No memories yet</li>';
    }

    // Thinking -- catch up from server on initial load
    if (data.thinking_history && thinkingHistory.length === 0 && data.thinking_history.length > 0) {
      for (const t of data.thinking_history) {
        thinkingHistory.push({
          text: t.thinking,
          decision: t.decision,
          tick: t.tick,
          time: new Date(t.time * 1000),
        });
      }
      renderThinkingHistory();
    }

    // Debug JSON
    document.getElementById('debug-json').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    console.error('Debug poll failed:', err);
  }
}

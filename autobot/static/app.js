/* Autobot — Frontend Application */

const API = '';
let sessionId = null;
let autoRunInterval = null;
let running = false;

// ===== DOM Refs =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const startScreen = $('#start-screen');
const simScreen = $('#sim-screen');
const useLlmToggle = $('#use-llm-toggle');

// Header
const scenarioTitle = $('#scenario-title');
const timeDisplay = $('#time-display');
const cycleDisplay = $('#cycle-display');
const stepBtn = $('#step-btn');
const run5Btn = $('#run5-btn');
const runBtn = $('#run-btn');
const stopBtn = $('#stop-btn');
const backBtn = $('#back-btn');

// Agent panel
const energyBar = $('#energy-bar');
const energyVal = $('#energy-val');
const repBar = $('#rep-bar');
const repVal = $('#rep-val');
const emotionsGrid = $('#emotions-grid');
const arousalVal = $('#arousal-val');
const valenceVal = $('#valence-val');
const salienceTags = $('#salience-tags');

// Feed
const eventFeed = $('#event-feed');

// World panel
const npcList = $('#npc-list');
const goalsList = $('#goals-list');
const projectsList = $('#projects-list');
const commitmentsList = $('#commitments-list');
const memoryList = $('#memory-list');

// ===== Start Screen =====
$$('.scenario-card').forEach(card => {
  card.addEventListener('click', () => {
    startScenario(card.dataset.scenario);
  });
});

backBtn.addEventListener('click', () => {
  stopAutoRun();
  sessionId = null;
  simScreen.classList.remove('active');
  startScreen.classList.add('active');
});

async function startScenario(scenario) {
  const resp = await fetch(`${API}/api/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scenario,
      use_llm: useLlmToggle.checked,
    }),
  });
  const data = await resp.json();
  sessionId = data.session_id;

  // Switch screens
  startScreen.classList.remove('active');
  simScreen.classList.add('active');

  // Set title
  const titles = {
    broken_promise: 'Broken Promise',
    hidden_betrayal: 'Hidden Betrayal',
    delayed_reward: 'Delayed Reward',
    competing_goals: 'Competing Goals',
    ambiguous_intent: 'Ambiguous Intent',
  };
  scenarioTitle.textContent = titles[scenario] || scenario;

  // Clear feed
  eventFeed.innerHTML = '';

  // Render initial state
  renderObservation(data.observation);
  renderEmotions({
    anxiety: 0, optimism: 0.5, irritability: 0,
    social_warmth: 0.5, avoidance_bias: 0, risk_tolerance: 0.5,
  });
  renderNpcs(data.npcs);
  cycleDisplay.textContent = 'Cycle 0';
}

// ===== Controls =====
stepBtn.addEventListener('click', () => doStep(1));
run5Btn.addEventListener('click', () => doStep(5));

runBtn.addEventListener('click', () => {
  if (running) return;
  running = true;
  runBtn.classList.add('hidden');
  stopBtn.classList.remove('hidden');
  autoRunInterval = setInterval(() => doStep(1), 800);
});

stopBtn.addEventListener('click', stopAutoRun);

function stopAutoRun() {
  running = false;
  if (autoRunInterval) clearInterval(autoRunInterval);
  autoRunInterval = null;
  stopBtn.classList.add('hidden');
  runBtn.classList.remove('hidden');
}

async function doStep(steps) {
  if (!sessionId) return;
  stepBtn.disabled = true;
  run5Btn.disabled = true;

  try {
    const resp = await fetch(`${API}/api/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        steps,
        use_llm: useLlmToggle.checked,
      }),
    });
    const data = await resp.json();

    for (const step of data.steps) {
      renderStep(step);
    }
  } catch (err) {
    console.error('Step failed:', err);
    stopAutoRun();
  } finally {
    stepBtn.disabled = false;
    run5Btn.disabled = false;
  }
}

// ===== Renderers =====

function renderStep(step) {
  timeDisplay.textContent = step.time;
  cycleDisplay.textContent = `Cycle ${step.cycle}`;

  renderObservation(step.observation);
  renderEmotions(step.emotions);
  renderAffect(step.affect);
  renderNpcs(step.npcs);
  renderGoals(step.goals_list);
  renderFeedEntry(step);
  renderMemories(step);
}

function renderObservation(obs) {
  if (!obs) return;
  const s = obs.self_state;
  if (!s) return;

  // Energy
  const ePct = Math.round(s.energy * 100);
  energyBar.style.width = ePct + '%';
  energyBar.className = 'bar energy-bar' +
    (s.energy < 0.15 ? ' critical' : s.energy < 0.3 ? ' low' : '');
  energyVal.textContent = s.energy.toFixed(2);

  // Reputation
  const rPct = Math.round(s.reputation * 100);
  repBar.style.width = rPct + '%';
  repBar.className = 'bar rep-bar' + (s.reputation < 0.3 ? ' low' : '');
  repVal.textContent = s.reputation.toFixed(2);

  // Projects
  if (obs.open_projects) {
    projectsList.innerHTML = obs.open_projects.map(p => {
      const pPct = Math.round(p.progress * 100);
      const cls = p.progress >= 1 ? 'complete' : p.overdue ? 'overdue' : '';
      return `
        <div class="project-card">
          <div class="project-name">${esc(p.name)}</div>
          <div class="project-bar-container">
            <div class="project-bar ${cls}" style="width:${pPct}%"></div>
          </div>
          <div class="project-meta">
            ${pPct}% — deadline: tick ${p.deadline}${p.overdue ? ' ⚠ OVERDUE' : ''}
          </div>
        </div>`;
    }).join('');
  }

  // Commitments
  if (obs.visible_commitments) {
    commitmentsList.innerHTML = obs.visible_commitments.map(c => {
      const cls = c.status === 'broken' ? 'broken' :
                  c.status === 'fulfilled' ? 'fulfilled' : '';
      return `
        <div class="commitment-item ${cls}">
          <div><b>${esc(c.deliverable)}</b></div>
          <div>→ ${esc(c.beneficiary)} | deadline: ${c.deadline} | ${c.status}</div>
        </div>`;
    }).join('');
  }
}

function renderEmotions(emo) {
  if (!emo) return;
  const items = [
    { name: 'Anxiety', val: emo.anxiety, warn: true },
    { name: 'Optimism', val: emo.optimism, good: true },
    { name: 'Irritability', val: emo.irritability, warn: true },
    { name: 'Warmth', val: emo.social_warmth, good: true },
    { name: 'Avoidance', val: emo.avoidance_bias, warn: true },
    { name: 'Risk Tol.', val: emo.risk_tolerance },
  ];

  emotionsGrid.innerHTML = items.map(i => {
    const cls = i.val > 0.6 ? (i.warn ? 'high' : 'low') :
                i.val > 0.35 ? 'mid' : (i.good ? 'high' : 'low');
    // Invert class logic: high anxiety = red, high optimism = green
    const level = i.val > 0.6 ? (i.warn ? 'high' : 'low') :
                  i.val < 0.2 ? (i.good ? 'high' : 'low') : 'mid';
    return `
      <div class="emotion-item ${level}">
        <span class="emo-name">${i.name}</span>
        <span class="emo-val">${i.val.toFixed(2)}</span>
      </div>`;
  }).join('');
}

function renderAffect(affect) {
  if (!affect) return;
  arousalVal.textContent = affect.arousal.toFixed(2);
  valenceVal.textContent = affect.valence.toFixed(2);

  arousalVal.style.color = affect.arousal > 0.5 ? 'var(--red)' :
                           affect.arousal > 0.2 ? 'var(--yellow)' : 'var(--text)';
  valenceVal.style.color = affect.valence < -0.3 ? 'var(--red)' :
                           affect.valence > 0.3 ? 'var(--green)' : 'var(--text)';

  const tags = affect.salience_tags || [];
  const negativeTags = new Set([
    'broken_promise', 'embarrassment', 'deadline_risk', 'betrayal',
    'trust_drop', 'reputation_threat', 'conflict', 'uncertainty',
  ]);
  const positiveTags = new Set(['praise', 'achievement', 'trust_gain']);

  salienceTags.innerHTML = tags.map(t => {
    const cls = negativeTags.has(t) ? 'negative' : positiveTags.has(t) ? 'positive' : '';
    return `<span class="tag ${cls}">${t}</span>`;
  }).join('');
}

function renderNpcs(npcs) {
  if (!npcs) return;
  npcList.innerHTML = npcs.map(n => {
    const tPct = Math.round(n.trust * 100);
    const tClass = n.trust >= 0.6 ? 'high' : n.trust >= 0.35 ? 'mid' : 'low';
    const barColor = n.trust >= 0.6 ? 'var(--green)' :
                     n.trust >= 0.35 ? 'var(--yellow)' : 'var(--red)';
    return `
      <div class="npc-card">
        <div class="npc-name">${esc(n.name)}</div>
        <div class="npc-stat">
          <span>Trust</span>
          <span class="val ${tClass}">${n.trust.toFixed(2)}</span>
        </div>
        <div class="trust-bar-container">
          <div class="trust-bar" style="width:${tPct}%; background:${barColor}"></div>
        </div>
        <div class="npc-stat" style="margin-top:4px">
          <span>Influence</span>
          <span class="val">${n.influence.toFixed(2)}</span>
        </div>
      </div>`;
  }).join('');
}

function renderGoals(goals) {
  if (!goals) return;
  goalsList.innerHTML = goals.map(g => {
    const cls = g.completed ? 'completed' : g.abandoned ? 'abandoned' : g.category;
    return `
      <div class="goal-item ${cls}">
        <div class="goal-desc">${esc(g.description)}</div>
        <div class="goal-meta">
          ${g.category} | priority: ${g.priority.toFixed(2)}
          ${g.failures > 0 ? ` | fails: ${g.failures}` : ''}
          ${g.completed ? ' ✓' : g.abandoned ? ' ✗' : ''}
        </div>
      </div>`;
  }).join('');
}

function renderMemories(step) {
  if (!step.memories) return;
  memoryList.innerHTML = `<pre style="font-size:0.72rem; color:var(--text-dim); white-space:pre-wrap">${esc(step.memories)}</pre>`;
}

function renderFeedEntry(step) {
  const div = document.createElement('div');
  div.className = 'feed-cycle flash';

  // Header with time + action
  const action = step.action;
  const actionName = action ? action.name : 'none';
  const actionArgs = action && action.args ?
    Object.entries(action.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ') : '';

  let header = `
    <div class="feed-cycle-header">
      <span class="feed-cycle-time">${esc(step.time)}</span>
      <span class="feed-action">${esc(actionName)}(${esc(actionArgs)})</span>
    </div>`;

  // Events
  let events = '';
  const significantTypes = new Set([
    'trust_change', 'reputation_change', 'commitment_broken', 'commitment_fulfilled',
    'deadline_passed', 'project_completed', 'action_failed', 'npc_response',
    'statement', 'information_received', 'information_refused', 'apology_given',
    'betrayal_exposed', 'accusation_made', 'praise_given', 'commitment_made',
    'project_progress',
  ]);

  for (const ev of (step.events || [])) {
    if (!significantTypes.has(ev.type)) continue;
    const { text, cls } = formatEvent(ev);
    events += `<div class="feed-event ${cls}">${text}</div>`;
  }

  if (step.new_episode) {
    events += `<div class="feed-event memory">📝 New episodic memory created</div>`;
  }

  div.innerHTML = header + events;
  eventFeed.prepend(div);

  // Keep feed manageable
  while (eventFeed.children.length > 100) {
    eventFeed.removeChild(eventFeed.lastChild);
  }
}

function formatEvent(ev) {
  const t = ev.type;
  if (t === 'trust_change') {
    const dir = ev.delta >= 0 ? 'trust-up' : 'trust-down';
    const sign = ev.delta >= 0 ? '+' : '';
    return {
      text: `Trust(${ev.npc}): ${sign}${ev.delta.toFixed(2)} — ${ev.reason || ''}`,
      cls: dir,
    };
  }
  if (t === 'reputation_change') {
    const dir = ev.delta >= 0 ? 'trust-up' : 'trust-down';
    return {
      text: `Reputation: ${(ev.old||0).toFixed(2)} → ${(ev.new||0).toFixed(2)}`,
      cls: dir,
    };
  }
  if (t === 'commitment_broken') {
    return { text: `💔 BROKEN: ${ev.deliverable} → ${ev.beneficiary}`, cls: 'important' };
  }
  if (t === 'commitment_fulfilled') {
    return { text: `✅ FULFILLED: ${ev.deliverable} → ${ev.beneficiary}`, cls: 'success' };
  }
  if (t === 'commitment_made') {
    return { text: `🤝 Committed: ${ev.deliverable} → ${ev.beneficiary}`, cls: '' };
  }
  if (t === 'deadline_passed') {
    return { text: `⏰ DEADLINE PASSED: ${ev.project}`, cls: 'important' };
  }
  if (t === 'project_completed') {
    return { text: `🏆 PROJECT COMPLETED: ${ev.project}`, cls: 'success' };
  }
  if (t === 'project_progress') {
    return { text: `📈 ${ev.project}: ${Math.round(ev.progress * 100)}%`, cls: '' };
  }
  if (t === 'action_failed') {
    return { text: `⚠ FAILED: ${ev.action} — ${ev.reason}`, cls: 'important' };
  }
  if (t === 'statement') {
    return { text: `💬 ${ev.speaker} → ${ev.target}: "${ev.text}"`, cls: '' };
  }
  if (t === 'npc_response') {
    return { text: `💬 ${ev.npc}: "${ev.text}"`, cls: '' };
  }
  if (t === 'information_received') {
    return { text: `ℹ️ ${ev.npc}: ${ev.text}`, cls: '' };
  }
  if (t === 'information_refused') {
    return { text: `🚫 ${ev.npc} refused: ${ev.text}`, cls: 'important' };
  }
  if (t === 'apology_given') {
    const repair = ev.repair ? ' (with repair)' : '';
    return { text: `🙏 Apologised to ${ev.target}${repair}`, cls: 'trust-up' };
  }
  if (t === 'betrayal_exposed') {
    return { text: `🗡️ BETRAYAL EXPOSED: ${ev.source}`, cls: 'important' };
  }
  if (t === 'accusation_made') {
    return { text: `⚡ Accused ${ev.target}: ${ev.claim}`, cls: 'important' };
  }
  if (t === 'praise_given') {
    return { text: `👏 Praised ${ev.target}`, cls: 'trust-up' };
  }
  return { text: `${t}: ${JSON.stringify(ev)}`, cls: '' };
}

function esc(str) {
  if (str === null || str === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(str);
  return d.innerHTML;
}

const API = 'http://localhost:8000'; // absolute URL — works from file:// and any other origin
let allTeams = [];

// ── Fetch team list ──────────────────────────
async function loadTeams() {
  try {
    const r = await fetch(`${API}/teams`);
    const d = await r.json();
    allTeams = d.teams;
  } catch (e) {
    console.warn('Could not load team list', e);
  }
}

// ── Autocomplete ─────────────────────────────
function setupAutocomplete(inputId, listId) {
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  let activeIdx = -1;

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    list.innerHTML = '';
    activeIdx = -1;
    if (!q) {
      list.classList.remove('open');
      return;
    }

    const matches = allTeams.filter(t => t.includes(q)).slice(0, 10);
    if (!matches.length) {
      list.classList.remove('open');
      return;
    }

    matches.forEach((team, i) => {
      const li = document.createElement('li');
      li.textContent = team;
      li.addEventListener('mousedown', () => {
        input.value = team;
        list.classList.remove('open');
      });
      list.appendChild(li);
    });
    list.classList.add('open');
  });

  input.addEventListener('keydown', e => {
    const items = list.querySelectorAll('li');
    if (e.key === 'ArrowDown') {
      activeIdx = Math.min(activeIdx + 1, items.length - 1);
    } else if (e.key === 'ArrowUp') {
      activeIdx = Math.max(activeIdx - 1, 0);
    } else if (e.key === 'Enter' && activeIdx >= 0) {
      input.value = items[activeIdx].textContent;
      list.classList.remove('open');
      e.preventDefault();
    } else if (e.key === 'Escape') {
      list.classList.remove('open');
    }
    items.forEach((el, i) => el.classList.toggle('active', i === activeIdx));
  });

  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !list.contains(e.target)) {
      list.classList.remove('open');
    }
  });
}

// ── Neutral toggle label ─────────────────────
const neutralToggle = document.getElementById('neutral-toggle');
const neutralLabel = document.getElementById('neutral-label');
if (neutralToggle && neutralLabel) {
  neutralToggle.addEventListener('change', () => {
    neutralLabel.textContent = neutralToggle.checked ? 'Neutral venue' : 'Home advantage ON';
  });
}

// ── Predict ──────────────────────────────────
const predictBtn = document.getElementById('predict-btn');
if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    const errorBox = document.getElementById('error-box');
    const card = document.getElementById('result-card');
    const homeTeam = document.getElementById('home-input').value.trim();
    const awayTeam = document.getElementById('away-input').value.trim();
    const competition = document.getElementById('competition').value;
    const isNeutral = neutralToggle.checked ? 1 : 0;

    errorBox.style.display = 'none';
    card.style.display = 'none';

    if (!homeTeam || !awayTeam) {
      showError('Please enter both team names.');
      return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting…';

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          competition,
          is_neutral: isNeutral
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed.');
      }

      const d = await res.json();
      renderResult(d);
    } catch (e) {
      showError(e.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = 'Predict Match';
    }
  });
}

function showError(msg) {
  const box = document.getElementById('error-box');
  if (box) {
    box.textContent = '⚠ ' + msg;
    box.style.display = 'block';
  }
}

function renderResult(d) {
  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

  document.getElementById('res-title').textContent = `${cap(d.home_team)} vs ${cap(d.away_team)}`;
  document.getElementById('res-comp').textContent = d.competition;
  document.getElementById('res-home-xg').textContent = d.home_xg.toFixed(2);
  document.getElementById('res-away-xg').textContent = d.away_xg.toFixed(2);
  document.getElementById('res-score1').textContent = d.most_likely_score;
  document.getElementById('res-score2').textContent = d.second_likely_score;

  document.getElementById('prob-home-lbl').textContent = `${cap(d.home_team)} Win`;
  document.getElementById('prob-away-lbl').textContent = `${cap(d.away_team)} Win`;
  document.getElementById('odds-home-lbl').textContent = `${cap(d.home_team)} Win`;
  document.getElementById('odds-away-lbl').textContent = `${cap(d.away_team)} Win`;

  const pHome = (d.p_home_win * 100).toFixed(1);
  const pDraw = (d.p_draw * 100).toFixed(1);
  const pAway = (d.p_away_win * 100).toFixed(1);

  document.getElementById('pct-home').textContent = pHome + '%';
  document.getElementById('pct-draw').textContent = pDraw + '%';
  document.getElementById('pct-away').textContent = pAway + '%';

  document.getElementById('odds-home').textContent = d.odds_home_win;
  document.getElementById('odds-draw').textContent = d.odds_draw;
  document.getElementById('odds-away').textContent = d.odds_away_win;

  const card = document.getElementById('result-card');
  card.style.display = 'block';

  // Animate bars after a tiny delay so CSS transition fires
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('bar-home').style.width = pHome + '%';
      document.getElementById('bar-draw').style.width = pDraw + '%';
      document.getElementById('bar-away').style.width = pAway + '%';
    }, 50);
  });
}

// Initialize
setupAutocomplete('home-input', 'home-list');
setupAutocomplete('away-input', 'away-list');
loadTeams();

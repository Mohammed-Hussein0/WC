// Automatically use current origin in production/server, or default to localhost:8000 if opened via file://
const API = (window.location.protocol === 'file:' || !window.location.origin || window.location.origin === 'null')
  ? 'http://localhost:8000'
  : window.location.origin;
let allTeams = [];

// ── Country Flag Mapping ──────────────────────
const TEAM_FLAGS = {
  "afghanistan": "af", "albania": "al", "algeria": "dz", "andorra": "ad", "angola": "ao",
  "antigua and barbuda": "ag", "argentina": "ar", "armenia": "am", "aruba": "aw",
  "australia": "au", "austria": "at", "azerbaijan": "az", "bahamas": "bs", "bahrain": "bh",
  "bangladesh": "bd", "barbados": "bb", "belarus": "by", "belgium": "be", "belize": "bz",
  "benin": "bj", "bermuda": "bm", "bhutan": "bt", "bolivia": "bo", "bosnia and herzegovina": "ba",
  "botswana": "bw", "brazil": "br", "brunei": "bn", "bulgaria": "bg", "burkina faso": "bf",
  "burundi": "bi", "cambodia": "kh", "cameroon": "cm", "canada": "ca", "cape verde": "cv",
  "central african republic": "cf", "chad": "td", "chile": "cl", "china pr": "cn", "china": "cn",
  "chinese taipei": "tw", "colombia": "co", "comoros": "km", "congo": "cg", "dr congo": "cd",
  "congo dr": "cd", "costa rica": "cr", "croatia": "hr", "cuba": "cu", "curacao": "cw",
  "cyprus": "cy", "czech republic": "cz", "czechia": "cz", "denmark": "dk", "djibouti": "dj",
  "dominica": "dm", "dominican republic": "do", "ecuador": "ec", "egypt": "eg",
  "el salvador": "sv", "england": "gb-eng", "equatorial guinea": "gq", "eritrea": "er",
  "estonia": "ee", "eswatini": "sz", "ethiopia": "et", "faroe islands": "fo", "fiji": "fj",
  "finland": "fi", "france": "fr", "gabon": "ga", "gambia": "gm", "georgia": "ge",
  "germany": "de", "ghana": "gh", "gibraltar": "gi", "greece": "gr", "grenada": "gd",
  "guatemala": "gt", "guinea": "gn", "guinea-bissau": "gw", "guyana": "gy", "haiti": "ht",
  "honduras": "hn", "hong kong": "hk", "hungary": "hu", "iceland": "is", "india": "in",
  "indonesia": "id", "iran": "ir", "iraq": "iq", "israel": "il", "italy": "it",
  "ivory coast": "ci", "côte d'ivoire": "ci", "jamaica": "jm", "japan": "jp", "jordan": "jo",
  "kazakhstan": "kz", "kenya": "ke", "kosovo": "xk", "kuwait": "kw", "kyrgyzstan": "kg",
  "laos": "la", "latvia": "lv", "lebanon": "lb", "lesotho": "ls", "liberia": "lr",
  "libya": "ly", "liechtenstein": "li", "lithuania": "lt", "luxembourg": "lu", "macau": "mo",
  "madagascar": "mg", "malawi": "mw", "malaysia": "my", "maldives": "mv", "mali": "ml",
  "malta": "mt", "mauritania": "mr", "mauritius": "mu", "mexico": "mx", "moldova": "md",
  "mongolia": "mn", "montenegro": "me", "montserrat": "ms", "morocco": "ma", "mozambique": "mz",
  "myanmar": "mm", "namibia": "na", "nepal": "np", "netherlands": "nl", "new caledonia": "nc",
  "new zealand": "nz", "nicaragua": "ni", "niger": "ne", "nigeria": "ng", "north macedonia": "mk",
  "northern ireland": "gb-nir", "norway": "no", "oman": "om", "pakistan": "pk", "palestine": "ps",
  "panama": "pa", "papua new guinea": "pg", "paraguay": "py", "peru": "pe", "philippines": "ph",
  "poland": "pl", "portugal": "pt", "puerto rico": "pr", "qatar": "qa", "republic of ireland": "ie",
  "ireland": "ie", "romania": "ro", "russia": "ru", "rwanda": "rw", "saint kitts and nevis": "kn",
  "saint lucia": "lc", "saint vincent and the grenadines": "vc", "samoa": "ws", "san marino": "sm",
  "sao tome and principe": "st", "saudi arabia": "sa", "scotland": "gb-sct", "senegal": "sn",
  "serbia": "rs", "seychelles": "sc", "sierra leone": "sl", "singapore": "sg", "slovakia": "sk",
  "slovenia": "si", "solomon islands": "sb", "somalia": "so", "south africa": "za",
  "south korea": "kr", "korea republic": "kr", "south sudan": "ss", "spain": "es", "sri lanka": "lk",
  "sudan": "sd", "suriname": "sr", "sweden": "se", "switzerland": "ch", "syria": "sy",
  "tahiti": "pf", "tajikistan": "tj", "tanzania": "tz", "thailand": "th", "timor-leste": "tl",
  "togo": "tg", "tonga": "to", "trinidad and tobago": "tt", "tunisia": "tn", "turkey": "tr",
  "türkiye": "tr", "turkmenistan": "tm", "uganda": "ug", "ukraine": "ua", "united arab emirates": "ae",
  "united states": "us", "usa": "us", "uruguay": "uy", "uzbekistan": "uz", "vanuatu": "vu",
  "venezuela": "ve", "vietnam": "vn", "wales": "gb-wls", "yemen": "ye", "zambia": "zm", "zimbabwe": "zw"
};

function getFlagImgHtml(teamName, isLarge = false) {
  if (!teamName) return '';
  const key = teamName.toLowerCase().trim();
  const code = TEAM_FLAGS[key];
  if (!code) return '';
  const cssClass = isLarge ? 'flag-img-lg' : 'flag-img';
  return `<img src="https://flagcdn.com/w40/${code}.png" class="${cssClass}" alt="" />`;
}

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
      const flagHtml = getFlagImgHtml(team);
      li.innerHTML = `${flagHtml} <span>${team}</span>`;
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
      const itemSpan = items[activeIdx].querySelector('span');
      input.value = itemSpan ? itemSpan.textContent : items[activeIdx].textContent;
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
          competition: "FIFA World Cup",
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

  const homeFlag = getFlagImgHtml(d.home_team, true);
  const awayFlag = getFlagImgHtml(d.away_team, true);

  document.getElementById('res-title').innerHTML = `${homeFlag} <span>${cap(d.home_team)}</span> <span style="color:var(--muted); font-size:14px; font-weight:600; margin:0 4px;">VS</span> ${awayFlag} <span>${cap(d.away_team)}</span>`;

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

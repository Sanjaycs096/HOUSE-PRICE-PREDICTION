// Predict page interactions: form submit, preview, Chart.js, GSAP micro-animations
(function(){
  const form = document.getElementById('predict-form');
  const tabularBtn = document.getElementById('tabular-predict-btn');
  const imageBtn = document.getElementById('image-predict-btn');
  const preview = document.getElementById('preview-img');
  const resultValue = document.getElementById('result-value');
  const resultMeta = document.getElementById('result-meta');
  const ctx = document.getElementById('price-chart').getContext('2d');
  let chart = null;

  function formatCurrency(v){
    return '₹' + Math.round(v).toLocaleString();
  }

  function makeTrend(base){
    // generate 12 month synthetic trend around base
    const points = [];
    for(let i=11;i>=0;i--){
      const noise = (Math.sin(i/3) + (Math.random()-0.5))*0.03;
      points.push(Math.max(0, base * (1 + noise)));
    }
    return points.reverse();
  }

  function updateChart(data){
    const labels = Array.from({length:12}, (_,i)=>`${i+1}m`);
    const dataset = {
      labels,
      datasets:[{label:'Estimated Price Trend',data,fill:true,borderColor:'#7b61ff',backgroundColor:'rgba(123,97,255,0.12)',tension:0.3}]
    };
    if (chart) {
      chart.data = dataset;
      chart.update();
    } else {
      chart = new Chart(ctx, {type:'line',data:dataset,options:{plugins:{legend:{display:false}},scales:{y:{ticks:{callback: v => '₹'+Math.round(v).toLocaleString()}}}}});
    }
  }

  function setLoading(isLoading){
    [tabularBtn, imageBtn].forEach(button => {
      if (!button) return;
      button.classList.toggle('loading', isLoading);
      button.disabled = isLoading;
    });
  }

  function showError(message){
    resultValue.textContent = message;
    resultMeta.textContent = '';
  }

  function getFormData(){
    const fd = new FormData(form);
    const tabularFields = Array.from(document.querySelectorAll('.tabular-field'));
    const tabularFilled = tabularFields.every(field => {
      const value = (field.value || '').toString().trim();
      return value !== '' && value !== '0';
    });
    return { fd, tabularFilled };
  }

  async function submitPrediction(endpoint){
    setLoading(true);
    resultValue.textContent = 'Calculating...';
    resultMeta.textContent = '';
    try{
      const { fd, tabularFilled } = getFormData();
      const imageFile = fd.get('image');
      if (endpoint === '/api/tabular' && !tabularFilled){
        showError('Please fill all required input fields before calculating the input prediction.');
        return;
      }
      if (endpoint === '/api/image' && !(imageFile && imageFile.name)){
        showError('Please upload an image before calculating the image prediction.');
        return;
      }
      // include CSRF header from meta tag
      const csrf = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
      const headers = csrf ? {'X-CSRF-Token': csrf} : {};
      const res = await fetch(endpoint, {method:'POST', body: fd, headers});
      const json = await res.json();
      if (!json.ok) throw new Error(json.error || 'Prediction failed');

      const price = json.result.price || json.result?.price || 0;
      if (window.gsap){
        gsap.fromTo(resultValue, {opacity:0,y:6},{opacity:1,y:0,duration:0.6});
      }
      resultValue.textContent = formatCurrency(price);
      resultMeta.textContent = endpoint === '/api/image' ? 'Source: Image model' : 'Source: Tabular model';

      const mPrice = document.getElementById('mortgage-price');
      if (mPrice) mPrice.value = Math.round(price);

      const trend = makeTrend(price);
      updateChart(trend);

      const record = {price: price, source: endpoint === '/api/image' ? 'image' : 'tabular', timestamp: new Date().toISOString()};
      const h = loadLocalHistory(); h.push(record); saveLocalHistory(h); renderHistory();

      try{ 
        const csrf2 = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
        const hdrs = Object.assign({'Content-Type':'application/json'}, csrf2 ? {'X-CSRF-Token': csrf2} : {});
        fetch('/api/history', {method:'POST', headers: hdrs, body: JSON.stringify(record)}).catch(()=>{});
      }catch(e){}
    }catch(err){
      showError(err.message || 'Error');
    }finally{
      setLoading(false);
    }
  }

  // History helpers (localStorage + server)
  const HISTORY_KEY = 'hp_history_v1';
  const historyListEl = document.getElementById('history-list');
  const clearBtn = document.getElementById('clear-history');

  function loadLocalHistory(){
    try{ return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }catch(e){return []}
  }

  function saveLocalHistory(items){
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(-50)));
  }

  function renderHistory(){
    const items = loadLocalHistory().slice().reverse();
    historyListEl.innerHTML = '';
    if (!items.length){ historyListEl.innerHTML = '<li class="muted">No saved predictions yet.</li>'; return }
    items.forEach(it =>{
      const li = document.createElement('li');
      li.className = 'history-item';
      const time = new Date(it.timestamp).toLocaleString();
      li.innerHTML = `<div class="h-left"><div class="h-price">${formatCurrency(it.price)}</div><div class="h-meta muted small">${it.source} • ${time}</div></div>`;
      li.addEventListener('click', ()=>{
        // populate result view from history
        resultValue.textContent = formatCurrency(it.price);
        resultMeta.textContent = 'Loaded from history';
        const trend = makeTrend(it.price);
        updateChart(trend);
      });
      historyListEl.appendChild(li);
    })
  }

  clearBtn?.addEventListener('click', ()=>{
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });


  form.addEventListener('change', (e)=>{
    if (e.target && e.target.id === 'image-input'){
      const f = e.target.files[0];
      if (!f) { preview.style.display='none'; preview.src=''; return; }
      const r = new FileReader();
      r.onload = ()=>{ preview.src = r.result; preview.style.display='block'; };
      r.readAsDataURL(f);
    }
  });

  tabularBtn?.addEventListener('click', (ev)=>{
    ev.preventDefault();
    submitPrediction('/api/tabular');
  });

  imageBtn?.addEventListener('click', (ev)=>{
    ev.preventDefault();
    submitPrediction('/api/image');
  });

  // initialize history on load
  renderHistory();

  // Mortgage calculator
  function calcMortgage(paymentAmount, downPct, annualRate, years){
    const principal = Math.max(0, paymentAmount * (1 - (downPct/100)));
    const monthlyRate = (annualRate/100)/12;
    const n = years * 12;
    if (monthlyRate === 0) return principal / n;
    const factor = Math.pow(1+monthlyRate, n);
    const monthly = principal * (monthlyRate * factor) / (factor - 1);
    return monthly;
  }

  const mortgageBtn = document.getElementById('mortgage-calc');
  const mortgageResult = document.getElementById('mortgage-result');
  const mortgageChartCtx = document.getElementById('mortgage-chart')?.getContext('2d');
  let mortgageChart = null;

  function updateMortgageChart(monthly, years){
    const n = years*12;
    const labels = Array.from({length:n}, (_,i)=> (i+1));
    const data = Array.from({length:n}, ()=> monthly);
    if (!mortgageChartCtx) return;
    if (mortgageChart){ mortgageChart.data = {labels, datasets:datasets(data)}; mortgageChart.update(); return; }
    mortgageChart = new Chart(mortgageChartCtx, {type:'bar', data:{labels, datasets:[{label:'Monthly payment', data, backgroundColor:'rgba(123,97,255,0.14)'}]}, options:{plugins:{legend:{display:false}},scales:{x:{display:false}, y:{ticks:{callback:v=>'₹'+Math.round(v).toLocaleString()}}}}});
  }

  function datasets(data){ return [{label:'Monthly payment', data, backgroundColor:'rgba(123,97,255,0.14)'}] }

  mortgageBtn?.addEventListener('click', (e)=>{
    e.preventDefault();
    const price = Number(document.getElementById('mortgage-price')?.value || 0);
    const down = Number(document.getElementById('mortgage-down')?.value || 20);
    const rate = Number(document.getElementById('mortgage-rate')?.value || 7);
    const term = Number(document.getElementById('mortgage-term')?.value || 20);
    if (price <= 0){ mortgageResult.textContent = 'Enter a valid property price.'; return; }
    const monthly = calcMortgage(price, down, rate, term);
    mortgageResult.textContent = `Monthly payment: ${formatCurrency(monthly)} (term ${term} years)`;
    updateMortgageChart(monthly, term);
  });

})();

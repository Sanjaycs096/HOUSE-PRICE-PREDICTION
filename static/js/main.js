// Initialize particles (if library available)
if (window.particlesJS) {
  particlesJS('particles-js',{
    particles:{
      number:{value:40,density:{enable:true,value_area:800}},
      color:{value:'#ffffff'},
      opacity:{value:0.06},
      size:{value:3},
      line_linked:{enable:true,opacity:0.03}
    }
  })
}

// Simple header animation
document.addEventListener('DOMContentLoaded', ()=>{
  if (window.gsap) gsap.from('.brand', {opacity:0,y:-8,duration:0.8});
});

// Fetch market summary and render Chart.js bar chart
async function renderMarketChart(){
  try{
    const res = await fetch('/api/market/summary');
    const j = await res.json();
    if (!j.ok) return;
    const data = j.summary;
    const labels = data.map(d=>d.location);
    const values = data.map(d=>Math.round(d.avg_price));
    const ctx = document.getElementById('market-chart');
    if (!ctx) return;
    new Chart(ctx, {type:'bar', data:{labels, datasets:[{label:'Avg Price',data:values,backgroundColor:'#7b61ff'}]}, options:{plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>'₹'+Math.round(v).toLocaleString()}}}}});
  }catch(e){console.warn('market chart failed', e)}
}

document.addEventListener('DOMContentLoaded', ()=>{ renderMarketChart(); });

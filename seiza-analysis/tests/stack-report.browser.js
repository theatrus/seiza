
const result=document.createElement('pre');result.id='qa-result';document.body.prepend(result);
try{
 const check=(value,message)=>{if(!value)throw Error(message)};
 check(document.querySelector('#selector button').getAttribute('aria-pressed')==='true','first stack selection');
 const host=document.getElementById('stack-0'),rows=()=>host.querySelector('[data-id=frames]').children.length;
 check(rows()===100,'bounded admitted rows');
 check(!host.querySelector('.ledger').open,'input records start collapsed');
 host.querySelector('.ledger').open=true;
 const charts=host.querySelector('[data-id=charts]');
 check(charts.querySelectorAll('svg')[0].querySelectorAll('circle').length===1000,'reference RMS remains missing');
 check(charts.querySelectorAll('svg')[1].querySelectorAll('circle').length===1001,'all admitted final sample decisions plotted');
 const metric=charts.querySelector('select');metric.value='gain';metric.dispatchEvent(new Event('change'));
 check(charts.querySelector('.chart-empty').textContent.includes('No recorded'),'missing normalization has explicit empty state');
 metric.value='rms';metric.dispatchEvent(new Event('change'));

 host.querySelector('[data-id=next]').click();check(rows()===100,'next page');
 const search=host.querySelector('[data-id=search]');search.value='fixture-1000';search.dispatchEvent(new Event('input'));check(rows()===1,'search reaches last frame');
 const toggle=host.querySelector('.frame-toggle');toggle.click();check(toggle.getAttribute('aria-expanded')==='true','expand details');
 document.querySelectorAll('#selector button')[1].click();document.querySelectorAll('#selector button')[0].click();check(search.value==='fixture-1000','state survives switching');
 window.dispatchEvent(new Event('beforeprint'));window.dispatchEvent(new Event('beforeprint'));
 check(rows()===1001,'all admitted rows print');check([...host.querySelectorAll('.report-disclosure,.analysis-method,.attempts')].every(d=>d.open),'all disclosures print expanded');check(host.querySelector('.attempts tbody').children.length===205,'all rejected rows print');check(document.getElementById('stack-2').children.length>0,'unvisited stack prints');
 window.dispatchEvent(new Event('afterprint'));check(rows()===1,'search restored after print');check(host.querySelector('.frame-toggle').getAttribute('aria-expanded')==='true','expanded detail restored');
 result.textContent='PASS: charts, missing measurements, selection, pagination, search, details, switching, all-stack print and state restoration';
}catch(error){result.textContent='FAIL: '+error.message}

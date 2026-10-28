'use strict';
const documentReport=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const entries=documentReport.stacks||[{label:'Completed stack',report:documentReport}];
const fmt=v=>v==null?'—':typeof v==='number'?new Intl.NumberFormat('en',Number.isInteger(v)?{maximumFractionDigits:0}:{maximumSignificantDigits:5}).format(v):String(v);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n};
let selected=0,printing=false;
$('title').textContent=documentReport.stacks?documentReport.title:'Completed stack';
document.title='Seiza · '+$('title').textContent;
$('subtitle').textContent=entries.length+' independent stack result(s)';
const views=entries.map((entry,i)=>{const host=el('section',null,'stack-view');host.setAttribute('aria-label',entry.label);host.id='stack-'+i;host.hidden=i!==0;$('views').append(host);return host});
function ensure(i){if(!views[i].children.length)renderReport(entries[i].report,views[i],i,entries[i].label)}
function select(i){selected=i;views.forEach((v,j)=>v.hidden=i!==j);[...$('selector').children].forEach((b,j)=>b.setAttribute('aria-pressed',String(i===j)));ensure(i)}
if(documentReport.stacks&&entries.length>1){
 $('overview').hidden=false;
 entries.forEach((entry,i)=>{const r=entry.report,known=(r.context?.frames||[]).filter(f=>Number.isFinite(f.exposure_seconds)&&f.exposure_seconds>0),unknown=r.accepted_frames-known.length;
 const row=el('tr');[entry.label,r.stage+' · '+r.width+' × '+r.height+' · '+r.channels+' component(s)',fmt(known.reduce((n,f)=>n+f.exposure_seconds,0))+' s'+(unknown?' · '+unknown+' unknown':''),r.accepted_frames+' / '+r.rejected_frames,r.support.map(s=>fmt(100*s.supported_pixels/s.total_pixels)+'%').join(' / '),r.support.map(s=>fmt(s.support_mean)).join(' / ')+(r.stage==='drizzled'?' area × weight':' observations'),r.support.map(s=>fmt(s.rejected_samples)).join(' / ')].forEach(v=>row.append(el('td',v)));$('comparison').append(row);
 const button=el('button',entry.label);button.setAttribute('aria-controls','stack-'+i);button.onclick=()=>select(i);$('selector').append(button);
 });
}else $('selector').hidden=true;
function theme(){const dark=document.documentElement.dataset.theme==='dark';$('theme').textContent=dark?'Light mode':'Dark mode';$('theme').setAttribute('aria-pressed',String(dark))}
$('theme').onclick=()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';try{localStorage.setItem('seiza-report-theme',dark?'dark':'light')}catch(_){}theme()};theme();
$('download').href=URL.createObjectURL(new Blob([document.getElementById('data').textContent],{type:'application/json'}));$('print').onclick=()=>window.print();
// Print every stack, including all filtered-out or paginated records. Restore interactive state afterwards.
let savedViews,savedOverview;
window.addEventListener('beforeprint',()=>{if(printing)return;printing=true;savedOverview=$('overview').open;$('overview').open=true;savedViews=views.map(v=>[...v.childNodes]);views.forEach((v,i)=>{renderReport(entries[i].report,v,i,entries[i].label);v.querySelectorAll('.report-disclosure,.analysis-method,.attempts').forEach(d=>d.open=true)})});
window.addEventListener('afterprint',()=>{if(!printing)return;printing=false;$('overview').open=savedOverview;views.forEach((v,i)=>v.replaceChildren(...savedViews[i]));select(selected)});
select(0);
function renderReport(r,host,index,label){
 host.replaceChildren(document.getElementById('stack-view').content.cloneNode(true));
 host.querySelector('.stack-title').textContent=label;
 const $=id=>host.querySelector('[data-id="'+id+'"]'),ctx=r.context;
const name=p=>String(p).replaceAll(String.fromCharCode(92),'/').split('/').pop();
$('description').textContent=({online:'Online integration',batch:'Final integration',drizzled:'Drizzle integration'})[r.stage]+(typeof r.provenance?.output?.path==='string'?' · '+name(r.provenance.output.path):'');
const frames=ctx?.frames?.length?ctx.frames:(r.final_samples||[]).map(()=>({source:null,calibration_index:null,weight:[],noise:[],mapping:null})),known=frames.filter(f=>Number.isFinite(f.exposure_seconds)&&f.exposure_seconds>0),unknown=r.accepted_frames-known.length;
for(const [label,value,note] of [['Output grid',r.width+' × '+r.height,r.channels+' channel(s)'],['Admitted frames',fmt(r.accepted_frames),'Includes the reference only when integrated'],['Rejected attempts',fmt(r.rejected_frames),'Admission outcomes, separate from sample rejection'],['Known exposure',fmt(known.reduce((s,f)=>s+f.exposure_seconds,0))+' s',unknown+' admitted exposure(s) unavailable']]) {
 const d=el('div',null,'stat');d.append(el('div',label,'stat-label'),el('div',value,'stat-value'),el('div',note,'stat-note'));$('summary').append(d);
}
renderStackCharts(r,$('charts'),frames);
$('support-note').textContent=r.stage==='drizzled'?'Accumulated drop area × frame weight; these values are not frame counts.':'Accepted observations per output sample, after this stage’s sample decisions.';
function mapPanel(map,c,title,stats){
 const box=el('div',null,'map');box.append(el('h3',title));const canvas=el('canvas');canvas.width=map.width;canvas.height=map.height;canvas.style.aspectRatio=r.width+'/'+r.height;canvas.setAttribute('role','img');
 const values=map.values.filter((_,i)=>i%map.channels===c),max=Math.max(0,...values);
 canvas.setAttribute('aria-label',title+'; block mean range 0 to '+fmt(max));
 const dc=canvas.getContext('2d'),pixels=dc.createImageData(map.width,map.height);
 values.forEach((v,i)=>{const t=max>0?v/max:0;pixels.data.set(t>0?[35+110*t,76+130*t,110+105*t,255]:[18,30,42,255],i*4)});dc.putImageData(pixels,0,0);
 box.append(canvas,el('p','Block means · 0 — '+fmt(max)));if(stats)box.append(el('p',stats));$('maps').append(box);
}
r.support.forEach((s,c)=>{mapPanel(r.support_map,c,'Channel '+(c+1)+' · support',fmt(100*s.supported_pixels/s.total_pixels)+'% supported · mean '+fmt(s.support_mean)+' · range '+fmt(s.support_minimum)+'–'+fmt(s.support_maximum));if(r.rejection_map)mapPanel(r.rejection_map,c,'Channel '+(c+1)+' · rejected observations',fmt(s.rejected_samples)+' rejected samples at this stage')});
let page=0;
function renderFrames(){
 $('frames').replaceChildren();let count=0;
 frames.forEach((f,i)=>{if(!(f.source||'in-memory input').toLowerCase().includes($('search').value.toLowerCase()))return;count++;if(!printing&&(count<=page*100||count>(page+1)*100))return;
 const tr=el('tr'),td=el('td',null,'file'),button=el('button',null,'frame-toggle');button.type='button';button.setAttribute('aria-expanded','false');button.setAttribute('aria-controls','frame-'+index+'-'+i);
 const chevron=el('span','›','frame-chevron');chevron.setAttribute('aria-hidden','true');button.append(chevron,el('span',f.source?name(f.source):'In-memory input '+(i+1)));
 const detail=el('div',null,'frame-detail');detail.id='frame-'+index+'-'+i;detail.hidden=true;detail.append(el('p',f.source||'Source path unavailable'),el('pre',JSON.stringify({calibration:ctx?.calibrations[f.calibration_index]??null,weight:f.weight,noise:f.noise,mapping:f.mapping},null,2)));
 button.onclick=()=>{detail.hidden=!detail.hidden;button.setAttribute('aria-expanded',String(!detail.hidden))};td.append(button,detail);tr.append(td);
 const final=r.final_samples?.[i];[f.calibration_index==null?null:f.calibration_index+1,f.exposure_seconds,f.registration_rms_pixels,final?.finite_samples,final?.integrated_samples].forEach(v=>tr.append(el('td',fmt(v))));$('frames').append(tr);
 });$('previous').disabled=page===0;$('next').disabled=(page+1)*100>=count;$('frame-note').textContent=(printing?count:Math.min(100,Math.max(0,count-page*100)))+' shown / '+count+' matching / '+frames.length+' recorded admitted inputs. '+(r.final_samples?'Final sample counts describe the reintegration grid.':'Final reintegration sample counts are unavailable for this result.');
}
$('search').oninput=()=>{page=0;renderFrames()};$('previous').onclick=()=>{page--;renderFrames()};$('next').onclick=()=>{page++;renderFrames()};renderFrames();
function details(label,value){const d=el('details');d.append(el('summary',label),el('pre',JSON.stringify(value,null,2)));$('record').append(d)}
if(ctx){$('record').append(el('p','Reference: '+(ctx.reference_image||'path unavailable')),el('p',ctx.reference_integrated?'Reference contributes to the stack.':'Reference defines geometry and does not contribute as a reference.'));details('Admission settings',ctx.options);details('Calibration assignments',ctx.calibrations);ctx.planning_notes.forEach(n=>$('record').append(el('p',n)))}else $('record').append(el('p','Stack context was not supplied. Input history and settings are unavailable.'));
if(r.provenance)details('Admission and output provenance',r.provenance);
const failed=Array.isArray(r.provenance?.frames)?r.provenance.frames.filter(f=>f&&typeof f==='object'&&f.disposition!=='accepted'):[];
if(failed.length){
 const section=el('details',null,'panel attempts');section.append(el('summary',failed.length+' unsuccessful admission attempts'));
 const label=el('label','Find an attempt ','attempts-search'),search=el('input');search.type='search';search.placeholder='Filename or reason';label.append(search);section.append(label);
 const scroll=el('div',null,'table-scroll');scroll.tabIndex=0;scroll.setAttribute('role','region');scroll.setAttribute('aria-label','Unsuccessful admission attempts');
 const table=el('table'),thead=el('thead'),header=el('tr');header.append(el('th','Source'),el('th','Reason'));thead.append(header);table.append(thead);const body=el('tbody');table.append(body);scroll.append(table);section.append(scroll);
 const foot=el('div',null,'ledger-end'),note=el('p'),controls=el('div',null,'paging'),prev=el('button','Previous'),next=el('button','Next');controls.append(prev,next);foot.append(note,controls);section.append(foot);let page=0;
 const draw=()=>{const matches=failed.filter(f=>(String(f.source?.path||'Unknown source')+' '+String(f.reason||f.disposition)).toLowerCase().includes(search.value.toLowerCase()));body.replaceChildren();(printing?matches:matches.slice(page*100,(page+1)*100)).forEach(f=>{const row=el('tr');row.append(el('td',f.source?.path||'Unknown source'),el('td',f.reason||f.disposition));body.append(row)});note.textContent=body.children.length+' shown / '+matches.length+' matching attempts';prev.disabled=page===0;next.disabled=(page+1)*100>=matches.length};
 search.oninput=()=>{page=0;draw()};prev.onclick=()=>{page--;draw()};next.onclick=()=>{page++;draw()};section.ontoggle=()=>{if(section.open)draw()};draw();host.querySelector('.ledger').after(section);
}
r.warnings.forEach(w=>$('warnings').append(el('li',w)));
if(r.accepted_frames<3)$('warnings').prepend(el('li','Only '+r.accepted_frames+' admitted input(s). Sample rejection and noise interpretation are limited for this sparse result.'));
host.append(el('p','Stacking '+r.stacking_version+' · output '+r.output_version+' · schema '+r.schema_version,'reference'));
}

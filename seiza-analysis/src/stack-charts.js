// Charts use only retained values unless an explicit rich-analysis result is present.
function renderStackCharts(r,host,frames){
 const basename=p=>String(p||'In-memory input').replaceAll('\\','/').split('/').pop();
 host.append(el('h2','Frame diagnostics','section-heading'),el('p','Alignment during admission and sample rejection in the final integration.','section-intro'));
 const section=el('div',null,'charts');section.style.marginBottom='24px';host.append(section);
 function panel(title,note){const p=el('section',null,'panel'),head=el('div',null,'panel-head'),controls=el('div',null,'controls'),plot=el('div',null,'chart'),foot=el('p',note,'panel-foot');head.append(el('h2',title));p.append(head,controls,plot,foot);section.append(p);return {p,controls,plot,foot};}
 function selectControl(p,label,items){const wrap=el('label',label+' '),select=el('select');for(const [value,text] of items){const o=el('option',text);o.value=value;select.append(o)}wrap.append(select);p.controls.append(wrap);return select;}
 function extent(rows,key){return rows.reduce((a,p)=>[Math.min(a[0],p[key]),Math.max(a[1],p[key])],[Infinity,-Infinity]);}
 function scatter(p,rows,ylabel,xlabel='Admitted input order',time=false){
  rows=rows.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
  if(!rows.length){p.plot.replaceChildren(el('p','No recorded measurements for this selection.','chart-empty'));return;}
  const [xmin,xmax]=extent(rows,'x'),[lo,hi]=extent(rows,'y'),pad=Math.max((hi-lo)*.12,Math.abs(hi)*.01,.001);
  const c=chart(p.plot,{xmin,xmax,ymin:lo>=0?Math.max(0,lo-pad):lo-pad,ymax:hi+pad,xlabel,ylabel,time,integer:!time});
  for(const row of rows)dot(c,row.x,row.y,'var(--teal)',row.label+'\n'+fmt(row.y));
 }
 const attempts=new Map((Array.isArray(r.provenance?.frames)?r.provenance.frames:[]).filter(f=>f?.disposition==='accepted').map(f=>[f.source?.path,f.diagnostics]));
 const mean=values=>{const finite=(values||[]).filter(Number.isFinite);return finite.length?finite.reduce((a,b)=>a+b,0)/finite.length:null};
 const admission=panel('Admission diagnostics','Recorded during admission. These measurements do not describe later full-resolution refinement; the reference has no independently measured registration RMS.');
 const metric=selectControl(admission,'Metric',[['rms','Registration RMS · sensor px'],['drift','Registration drift · sensor px'],['rotation','Rotation · degrees'],['gain','Mean normalization gain']]);
 const drawAdmission=()=>scatter(admission,frames.map((f,i)=>{const d=attempts.get(f.source),t=f.mapping?.transform;return {x:i+1,label:basename(f.source),y:({rms:f.registration_rms_pixels,drift:d?.registration_drift_pixels,rotation:t?.rotation_radians==null?null:t.rotation_radians*180/Math.PI,gain:mean(f.mapping?.normalization?.gains)})[metric.value]}}),metric.selectedOptions[0].textContent);
 metric.onchange=drawAdmission;drawAdmission();
 const final=panel('Final sample decisions','Rejected finite samples / finite samples for each admitted input. Missing counts and inputs with no finite samples are omitted. This is not a frame-quality score.');
 scatter(final,frames.map((f,i)=>{const s=r.final_samples?.[i];return {x:i+1,label:basename(f.source),y:s?.finite_samples>0?100*(s.finite_samples-s.integrated_samples)/s.finite_samples:null}}),'Rejected finite samples · %');
 const a=r.provenance?.reporting?.analysis;
 if(!a)return;
 host.append(el('h2','Image quality & integration depth','section-heading'),el('p','Additional source measurements · independent of final stack admission','section-intro'));
 const note=el('details',null,'analysis-method');note.append(el('summary','How these measurements are calculated'),el('p','Additional pass over supplied source images, without applying stack calibration. CFA components use separate native lattices without interpolation; RGB retains prior processing correlations. Curves use sampled equal-weight diagnostics; they do not measure the final stack’s noise or SNR.'),el('p','Analysis pass: '+fmt(a.timings?.total_seconds)+' s · image reads '+fmt(a.timings?.read_seconds)+' s · measurements '+fmt(a.timings?.measurement_seconds)+' s · registration '+fmt(a.timings?.registration_seconds)+' s'));host.append(note);
 const rich=el('div',null,'charts');rich.style.marginBottom='24px';host.append(rich);
 const quality=panel('Source frame measurements','Native component measurements; compare within one pointing, filter and layout. CFA sizes use original-image pixels; sparse sampling can bias small stars.'),depth=panel('Sampled integration depth','Split-noise measurement versus the ideal 1/√N comparison. Shared systematic errors cancel; this does not reproduce production weighting or rejection.');rich.append(quality.p,depth.p);
 const groups=Array.isArray(a.groups)?a.groups:[],groupItems=groups.map((g,i)=>[String(i),(g.filter??'Unknown filter')+(g.component?' · '+g.image_kind.toUpperCase()+' / '+(g.component==='G_red'?'G (red row)':g.component==='G_blue'?'G (blue row)':g.component):'')]);
 const group=selectControl(quality,'Filter / component',groupItems),health=selectControl(quality,'Metric',[['median_hfr','HFR · image px'],['median_fwhm','FWHM · image px'],['median_eccentricity','Eccentricity'],['star_count','Star count'],['median_star_snr','Stellar peak SNR'],['background','Native background'],['noise','Native noise']]),order=selectControl(quality,'Order',[['order','Input order'],['time','Capture time · UTC']]);
 const axis=selectControl(depth,'Horizontal axis',[['frame_count','Frames'],['integration_seconds','Known integration · seconds']]);
 function drawRich(){
  const g=groups[Number(group.value)];
  if(!g){quality.plot.replaceChildren(el('p','No successfully measured filter groups.','chart-empty'));depth.plot.replaceChildren(el('p','No depth measurements.','chart-empty'));return;}
  const timed=order.value==='time';
  scatter(quality,(a.frames||[]).filter(f=>(f.filter??'').toLowerCase()===(g.filter??'').toLowerCase()&&(f.sampling?.image_kind??null)===(g.image_kind??null)&&(f.sampling?.component??null)===(g.component??null)).map((f,i)=>({f,x:timed?(f.capture_time?Date.parse(/[Zz]|[+-]\d\d:\d\d$/.test(f.capture_time)?f.capture_time:f.capture_time+'Z'):NaN):i+1,y:f[health.value],label:basename(f.path)})),health.selectedOptions[0].textContent,timed?'Capture time · UTC':'Analysis input order',timed);
  const points=(g.curve||[]).filter(p=>Number.isFinite(p.noise)&&Number.isFinite(p[axis.value]));
  if(!points.length){depth.plot.replaceChildren(el('p','Insufficient common samples or frames for a noise curve.','chart-empty'));return;}
  const ymax=points.reduce((m,p)=>Math.max(m,p.noise,Number.isFinite(p.ideal_noise)?p.ideal_noise:0),0)*1.12;
  const c=chart(depth.plot,{xmin:0,xmax:points.reduce((m,p)=>Math.max(m,p[axis.value]),1),ymin:0,ymax:Math.max(1e-20,ymax),xlabel:axis.selectedOptions[0].textContent,ylabel:'Noise · reference units',integer:axis.value==='frame_count'});
  for(const [key,color,dash] of [['noise','var(--blue)',''],['ideal_noise','var(--comparison)','5 4']])c.el('polyline',{points:points.filter(p=>Number.isFinite(p[key])).map(p=>`${c.x(p[axis.value])},${c.y(p[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dash});
  for(const p of points)dot(c,p[axis.value],p.noise,'var(--blue)',`N=${p.frame_count} · noise ${fmt(p.noise)}`);
  const t=g.noise_trend;depth.foot.textContent='Solid: measured split noise. Dashed: ideal 1/√N, anchored at two frames. '+(t?'Trend exponent '+fmt(t.exponent)+'; descriptive R² '+fmt(t.r_squared)+'. ':'')+'Diagnostic of source inputs; not final-stack noise. '+g.common_patches+' common patches. '+(g.warnings||[]).join(' ');
 }
 group.onchange=health.onchange=order.onchange=axis.onchange=drawRich;drawRich();
 const failures=(a.frames||[]).filter(f=>f.measurement_status!=='measured'||f.depth_status!=='included');
 if(failures.length){const d=el('details',null,'panel'),summary=el('summary','Analysis omissions · '+failures.length+' input(s)');d.append(summary,el('pre',JSON.stringify(failures.map(f=>({path:f.path,measurement_status:f.measurement_status,error:f.error,depth_status:f.depth_status,depth_error:f.depth_error})),null,2)));host.append(d);}
}

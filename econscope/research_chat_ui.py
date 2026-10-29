"""Authored chat-first shell; no owner HTML scripts/assets are copied or run."""
from html import escape
import json


def default_studies():
    rights={'local_processing':'PERMITTED','report_export':'PERMITTED'}
    common={'schema':'societybox-engine-study-v1', 'execution':'EXPLICIT_INDEPENDENT_EXACT',
            'source':{'id':'authored-synthetic-example','license':'CC0-1.0','kind':'SYNTHETIC',
                      'basis':'Invented demonstration coefficients, not observed economic data.'},'rights':rights}
    studies = {
        'input_output':{**common,'family':'input_output','question':'Conditional production changes under declared fixed coefficients?',
            'request':{'engine':'SocietyBoxExact','model':'static_leontief','parameters':{
                'schema':'societybox-input-output-conditional-v1','sectors':['steel','machinery'],
                'coefficient_matrix':[['0','1/2'],['1/4','0']],'final_demand_change':['100','0'],
                'unit':'declared million currency units (fixed-price scenario)',
                'source_declared_sha256':'a'*64,'rights':rights}}},
        'survey_replicate':{**common,'family':'survey_replicate','question':'Mean and variance conditional on declared replicate weights?',
            'request':{'engine':'SocietyBoxExact','model':'declared_replicate_mean','parameters':{
                'schema':'societybox-survey-replicate-mean-v1','values':['0','10'],'base_weights':['1','1'],
                'replicate_weights':[['2','0'],['0','2']],'method':'BRR_DECLARED',
                'unit':'declared index units','source_declared_sha256':'c'*64,'rights':rights}}},
    }
    from importlib.resources import files
    from .engine_study import _parse, study_families
    from .research_vintages import example_parameters
    vintage = example_parameters()
    studies['vintage_comparison'] = {
        'schema': 'societybox-engine-study-v1', 'family': 'vintage_comparison',
        'question': 'Which declared revisions were available at the two cutoffs?',
        'source': {**vintage['ledger']['source'], 'kind': 'SYNTHETIC'},
        'rights': vintage['ledger']['rights'], 'execution': 'EXPLICIT_INDEPENDENT_EXACT',
        'request': {'engine': 'SocietyBoxExact', 'model': 'declared_revision_comparison', 'parameters': vintage}}
    raw=files('econscope').joinpath('resources/research-studies.v1.json').read_bytes()
    if len(raw)>1024*1024: raise ValueError('CHAT_TEMPLATE_BUDGET')
    studies.update(_parse(raw))
    from .forecast_vintages import example_parameters as forecast_example
    forecast=forecast_example()
    studies['forecast_validation'] = {
        'schema': 'societybox-engine-study-v1', 'family': 'forecast_validation',
        'question': 'Conditional forecasts using the revisions declared available at each origin?',
        'source': {**forecast['revision_history']['ledger']['source'], 'kind': 'SYNTHETIC'},
        'rights': forecast['screen']['rights'], 'execution': 'EXPLICIT_PINNED_HIGHS',
        'request': {'engine': 'HiGHS+ExactForecast', 'model': 'declared_originwise_lag_forecast', 'parameters': forecast}}
    from .fixed_effects import example_parameters as fixed_effects_example
    fixed=fixed_effects_example()
    studies['fixed_effects_regression'] = {
        'schema':'societybox-engine-study-v1','family':'fixed_effects_regression',
        'question':'Conditional within-group association under declared analytic weights?',
        'source':{'id':'authored-fixed-effects','license':'CC0-1.0','kind':'SYNTHETIC',
                  'basis':'Invented fixed effects and residuals; no observed causal effect.'},
        'rights':fixed['rights'],'execution':'EXPLICIT_PINNED_HIGHS',
        'request':{'engine':'HiGHS+ExactFixedEffects','model':'declared_weighted_fixed_effects','parameters':fixed}}
    if set(studies)!=set(study_families()): raise ValueError('CHAT_TEMPLATE_REGISTRY_MISMATCH')
    return studies


_KEYS=('title','navigation','conversation','inspector','question','send','model','review','confirm','run',
       'versions','runs','previous','next','source','result','export','file','draft','ready','working','locked',
       'no_llm','rights','approval','no_result','unknown','edit','saved','key','connect','error','report','examples')
_TEXT={
 'en':('Research workspace','Research navigation','Research conversation','Evidence Inspector','Research question or follow-up',
       'Save message','Declared model and data','Save version and review','I reviewed these inputs, rights and the displayed digest.',
       'Approve and run reviewed version','Model versions','Calculation history','Previous','Next','Source and vintage','Conditional result',
       'Download verified bundle','Attach study JSON','Draft changed. Review this version before running.','Reviewed; execution not authorized.',
       'Working…','Open the private startup URL or enter its local access key.',
       'NO_LLM · Messages are saved without AI reasoning. Calculations use declared inputs. No external transfer.',
       'Rights are declarations, not source authentication. Release vintage: NOT_ESTABLISHED.',
       'Execution approval','No calculated result selected.','Source, design and causal validity remain NOT_ESTABLISHED.',
       'Edit study JSON','Saved locally','Local access key','Connect','Request rejected. Refresh state and review again.',
       'Verified research report','Synthetic examples; replace declarations for your own study.'),
 'ko':('연구 워크스페이스','연구 탐색','연구 대화','Evidence Inspector · 증거 검토','연구 질문 또는 후속 질문',
       '메시지 저장','선언된 모형과 자료','버전 저장·검토','입력·권리와 표시된 digest를 검토했습니다.',
       '검토한 버전 승인·계산','모형 버전','계산 이력','이전','다음','출처와 빈티지','조건부 계산 결과',
       '검증된 재현 묶음 다운로드','연구 JSON 첨부','초안이 변경되었습니다. 이 버전을 검토한 후 실행하세요.',
       '검토됨 · 실행 승인 전','처리 중…','시작 시 표시된 비공개 URL을 열거나 로컬 접근 키를 입력하세요.',
       'NO_LLM · AI 추론 없이 메시지를 저장합니다. 선언된 입력으로 계산하며 외부 전송은 없습니다.',
       '권리는 선언이며 출처 인증이 아닙니다. 공표 빈티지: NOT_ESTABLISHED.',
       '실행 승인','선택한 계산 결과가 없습니다.','출처·표본설계·인과적 타당성은 NOT_ESTABLISHED입니다.',
       '연구 JSON 편집','로컬 저장됨','로컬 접근 키','연결','요청이 거부되었습니다. 상태를 새로 읽고 다시 검토하세요.',
       '검증된 연구 보고서','합성 예제입니다. 실제 연구에는 자료와 권리 선언을 교체하세요.'),
 'zh-Hans':('研究工作空间','研究导航','研究对话','Evidence Inspector · 证据审查','研究问题或后续问题',
       '保存消息','声明的模型与数据','保存版本并审查','我已审查输入、权利及所示摘要。','批准并计算已审查版本',
       '模型版本','计算历史','上一页','下一页','来源与历史版本','条件计算结果','下载验证后的复现包','附加研究JSON',
       '草稿已修改。请先审查新版本。','已审查；尚未授权执行。','处理中…','请打开启动时的私有链接或输入本地访问密钥。',
       'NO_LLM · 仅保存消息，不执行AI推理。计算使用声明的输入，无外部传输。',
       '权利为声明，不代表来源认证。发布历史版本：NOT_ESTABLISHED。','执行批准','未选择计算结果。',
       '来源、抽样设计与因果有效性仍为NOT_ESTABLISHED。','编辑研究JSON','已保存到本地','本地访问密钥','连接',
       '请求被拒绝。请刷新状态并重新审查。','已验证的研究报告','合成示例；实际研究请替换数据及权利声明。'),
 'ja':('研究ワークスペース','研究ナビゲーション','研究対話','Evidence Inspector · 証拠確認','研究上の問い・追加の問い',
       'メッセージを保存','申告されたモデルとデータ','バージョンを保存・確認','入力・権利と表示ダイジェストを確認しました。',
       '確認済みバージョンを承認・計算','モデルのバージョン','計算履歴','前へ','次へ','出典とビンテージ','条件付き計算結果',
       '検証済み再現バンドルを取得','研究JSONを添付','下書きが変更されました。実行前に再確認してください。','確認済み・実行承認前',
       '処理中…','起動時の非公開URLを開くか、ローカルアクセスキーを入力してください。',
       'NO_LLM · AI推論なしでメッセージを保存。申告入力で計算し、外部転送はありません。',
       '権利は申告であり出典認証ではありません。公開ビンテージ：NOT_ESTABLISHED。','実行承認','計算結果が選択されていません。',
       '出典・標本設計・因果的妥当性はNOT_ESTABLISHEDです。','研究JSONを編集','ローカル保存済み','ローカルアクセスキー',
       '接続','リクエストが拒否されました。状態を更新し再確認してください。','検証済み研究レポート','合成例です。実際の研究では入力と権利の申告を変更してください。'),
}

STYLE=r"""
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#f6f7f5;color:#25382f;font:15px/1.5 system-ui,sans-serif}
header{padding:.8rem 1.2rem;background:white;border-bottom:1px solid #d9dfd9}h1{font-size:1.15rem;margin:0}h2{font-size:1rem}h3{font-size:.95rem}
.layout{display:grid;grid-template-columns:minmax(12rem,1fr) minmax(0,3fr) minmax(15rem,1.4fr);gap:1rem;padding:1rem;max-width:1800px;margin:auto}
nav,main,aside{min-width:0}nav,aside{background:#eef2ed;border:1px solid #d9dfd9;border-radius:.6rem;padding:.8rem;align-self:start}
article,.card,form{background:white;border:1px solid #d9dfd9;border-radius:.6rem;padding:.8rem;margin-bottom:.8rem;min-width:0}
.researcher{margin-left:1.4rem;border-left:3px solid #446957}.proposal{border-left:3px solid #8a7853}.calculation{border-left:3px solid #456e86}
p,pre,code,summary,label,button{overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:.8rem;max-height:24rem;overflow:auto}p{white-space:pre-wrap}
table{width:100%;border-collapse:collapse}th,td{text-align:start;padding:.45rem;border-bottom:1px solid #d9dfd9;overflow-wrap:anywhere}.table-scroll{overflow-x:auto;max-width:100%}
textarea{width:100%;max-width:100%;resize:vertical;padding:.6rem;font:inherit;border:1px solid #aebdb2;border-radius:.3rem}#spec{font:13px/1.5 ui-monospace,monospace}
button,select,input{font:inherit;max-width:100%}button,select{padding:.45rem .6rem;border:1px solid #9aafa2;border-radius:.3rem;background:white;color:#25382f;margin:.2rem}
button{cursor:pointer}button.primary{background:#315e48;color:white}button:disabled{cursor:default;opacity:.5}button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #2466c2;outline-offset:2px}
label{display:block;margin:.4rem 0}label.check{display:flex;gap:.5rem;align-items:flex-start}.check input{flex-shrink:0;margin-top:.35rem}small,.muted{color:#53675b}#status{min-height:1.5rem}#status[data-error=true]{color:#9a261d;font-weight:600}
nav button{display:block;text-align:left;width:100%;margin:.3rem 0}nav button[aria-current=true]{border:2px solid #315e48}details{min-width:0}summary{cursor:pointer}iframe{display:block;width:100%;height:36rem;border:1px solid #aebdb2;background:white}
.pager{display:flex;align-items:center;flex-wrap:wrap;gap:.25rem}.bar{display:flex;align-items:center;justify-content:space-between;gap:.5rem;flex-wrap:wrap}[hidden]{display:none!important}#messages{min-height:6rem}#source-info{font-size:.85rem}#access{max-width:40rem;margin:1rem auto}#inspector-records article{padding:.5rem}.json-file{margin:.5rem 0}
#messages{max-height:24rem;overflow-y:auto;overscroll-behavior:contain}nav details{max-height:75vh;overflow-y:auto}
@media(max-width:70rem){.layout{grid-template-columns:minmax(11rem,1fr) minmax(0,3fr)}aside{grid-column:1/-1}#inspector-records{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem}}
@media(max-width:42rem){.layout{display:block;padding:.5rem}nav,aside{margin-bottom:.8rem}nav details:not([open]){margin:0}header{padding:.7rem}button{min-height:2.6rem}#inspector-records{display:block}iframe{height:30rem}.researcher{margin-left:.4rem}}
"""

SCRIPT=r"""
(() => {
'use strict';
const $=id=>document.getElementById(id), t=JSON.parse($('labels').textContent);
const locale=document.documentElement.lang;
let token=location.hash.slice(1), state=null, templates={}, capabilities={families:[]}, selected=null, selectedRun=null;
let dirty=true, busy=false, messageOffset=null, versionOffset=null, runOffset=null, evidenceOffset=0, requestNumber=0;
let activeJob=null,viewedJob=null,jobOffset=0,jobTotal=0,pollTimer=null;
let localModel=null,localReview=null,localTurn=null,localSending=false;
const terminal=job=>['SUCCEEDED','FAILED','CANCELLED','EXPIRED','RECOVERY_REQUIRED'].includes(job.status);
history.replaceState(null,'',location.pathname+location.search);
const text=(node,value)=>{node.textContent=String(value);return node;};
const element=(tag,value,cls)=>{const n=document.createElement(tag);if(value!==undefined)text(n,value);if(cls)n.className=cls;return n;};
const status=(value,error=false)=>{text($('status'),value);$('status').dataset.error=String(error);};
async function api(path,data){
  const response=await fetch(path,{method:data===undefined?'GET':'POST',credentials:'omit',cache:'no-store',redirect:'error',
    headers:{Authorization:'Bearer '+token,...(data===undefined?{}:{'Content-Type':'application/json'})},body:data===undefined?undefined:JSON.stringify(data)});
  if(!response.ok){const error=new Error('REQUEST_REJECTED');error.status=response.status;throw error;}
  return response.json();
}
function controls(){
  document.querySelectorAll('button,input,textarea,select').forEach(n=>{n.disabled=busy||!state;});
  $('access-key').disabled=busy;$('connect').disabled=busy;
  const current=selected&&state&&selected.id===state.current_version_id;
  const alreadyRun=current&&state.runs.some(r=>r.version_id===selected.id);
  const blocked=spec=>{
    const policy=capabilities.families.find(row=>row.family===spec?.family);
    return !policy||policy.blockers.length>0||(spec?.request?.engine==='Pyomo+HiGHS'&&capabilities.sources==='NOT_CONFIGURED');
  };
  let draft=null;try{draft=JSON.parse($('spec').value);}catch(error){}
  $('review-model').disabled=busy||!state||blocked(draft);
  $('approve').disabled=busy||dirty||!current||alreadyRun||Boolean(activeJob)||capabilities.recovery_required||blocked(selected?.spec);
  $('execute').disabled=$('approve').disabled||!$('approve').checked;
  $('messages-prev').disabled=busy||!state||state.message_offset===0;
  $('messages-next').disabled=busy||!state||state.message_offset+state.messages.length>=state.message_count;
  for(const [key,prefix] of [['versions','version'],['runs','run']]){
    $(key+'-prev').disabled=busy||!state||state[prefix+'_offset']===0;
    $(key+'-next').disabled=busy||!state||state[prefix+'_offset']+state[key].length>=state[prefix+'_count'];
  }
  $('download').disabled=busy||!selectedRun;
  $('evidence-prev').disabled=busy||!selectedRun||evidenceOffset===0;
  $('evidence-next').disabled=busy||!selectedRun||evidenceOffset+30>=Number($('evidence-count').dataset.total||0);
  $('cancel-job').disabled=!activeJob||terminal(activeJob)||activeJob.status==='CANCEL_REQUESTED';
  $('recover-job').disabled=busy||!viewedJob||viewedJob.status!=='RECOVERY_REQUIRED';
  $('recovery-note').hidden=!capabilities.recovery_required&&viewedJob?.status!=='RECOVERY_REQUIRED';
  $('job-history-prev').disabled=busy||jobOffset===0;
  $('job-history-next').disabled=busy||jobOffset+40>=jobTotal;
  const configured=localModel?.mode==='LOCAL_LLM_CONFIGURED_NOT_CONNECTED';
  const currentReview=localReview&&state&&localReview.expected_revision===state.revision;
  $('llm-review-button').disabled=busy||!state||!configured||!$('llm-selected-text').value.trim();
  $('llm-confirm').disabled=busy||!currentReview||!configured||Boolean(localTurn);
  $('llm-begin').disabled=busy||!currentReview||!$('llm-confirm').checked||Boolean(localTurn);
  $('llm-send').disabled=busy||!currentReview||!localTurn||localTurn.status!=='AUTHORIZED_NOT_SENT';
  if(localSending)$('local-llm').querySelectorAll('button,input,textarea,select').forEach(n=>{n.disabled=true;});
}
async function work(action){
  if(busy)return;busy=true;controls();$('workspace').setAttribute('aria-busy','true');status(t.working);
  try{await action();status(t.saved);}catch(error){status(t.error,true);$('approve').checked=false;}
  finally{busy=false;$('workspace').setAttribute('aria-busy','false');controls();}
}
function changed(){dirty=true;$('approve').checked=false;status(t.draft);controls();}
function selectVersion(version){
  selected=version;dirty=false;$('approve').checked=false;
  clearResult();
  $('spec').value=JSON.stringify(version.spec,null,2);$('family').value=version.spec.family;
  text($('review-digest'),version.id+' · '+version.review_sha256);
  text($('source-info'),JSON.stringify({source:version.spec.source,rights:version.spec.rights,
    declared_source_sha256:version.spec.request.parameters?.source_declared_sha256||'NOT_ESTABLISHED',
    release_vintage:'NOT_ESTABLISHED'},null,2));
  document.querySelectorAll('[data-version]').forEach(n=>n.setAttribute('aria-current',String(n.dataset.version===version.id)));
  status(t.ready);controls();
}
function render(next){
  if(state&&next.revision<state.revision)return;
  const draft=$('spec').value, wasDirty=dirty;
  state=next;
  const messages=$('messages');messages.replaceChildren();
  for(const message of state.messages){
    const article=element('article',undefined,message.kind);
    article.append(element('small',message.kind==='researcher'?t.question:message.kind==='proposal'?t.review:'NO_LLM · '+t.result),element('p',message.text));
    if(message.run_id){const b=element('button',t.result);b.type='button';b.onclick=()=>work(()=>showRun(message.run_id));article.append(b);}
    messages.append(article);
  }
  text($('messages-count'),(state.message_count?state.message_offset+1:0)+'–'+(state.message_offset+state.messages.length)+' / '+state.message_count);
  for(const key of ['versions','runs']){
    const node=$(key);node.replaceChildren();
    for(const row of state[key]){
      const b=element('button',key==='versions'?row.id+' · '+row.spec.question:row.version_id+' · '+row.id.slice(0,8));b.type='button';
      if(key==='versions'){b.dataset.version=row.id;b.onclick=()=>selectVersion(row);}
      else b.onclick=()=>work(()=>showRun(row.id));
      node.append(b);
    }
    text($(key+'-count'),String(state[key.slice(0,-1)+'_count']));
  }
  if(state.versions.length){const v=state.versions.find(v=>selected&&v.id===selected.id)||state.versions[state.versions.length-1];selectVersion(v);}
  if(wasDirty){$('spec').value=draft;dirty=true;}
  $('access').hidden=true;controls();
}
async function refresh(){
  const q=new URLSearchParams();
  for(const [key,value] of [['messages',messageOffset],['versions',versionOffset],['runs',runOffset]])if(value!==null)q.set(key,value);
  render(await api('/api/state'+(q.size?'?'+q:'')));
}
function clearResult(){
  selectedRun=null;$('inspector-records').replaceChildren();$('report').hidden=true;$('report').removeAttribute('srcdoc');
  $('result-values').replaceChildren();text($('source-info'),'');
  text($('execution-status'),t.no_result);
  text($('result-summary'),t.no_result);text($('evidence-count'),'0');$('evidence-count').dataset.total='0';controls();
}
async function showRun(id){
  const request=++requestNumber;clearResult();
  const details=await api('/api/run/'+encodeURIComponent(id)+'?locale='+locale+'&offset='+evidenceOffset);
  if(request!==requestNumber)return;
  selectedRun=id;text($('result-summary'),details.verification.status+' · '+details.authorization+'\n'+JSON.stringify(details.result,null,2));
  text($('source-info'),JSON.stringify(details.source_context,null,2));
  text($('execution-status'),details.verification.status+' · native_execution: '+details.verification.native_execution+' · '+details.verification.native_parity);
  const r=details.result, table=element('table'), body=element('tbody');
  const head=element('thead'),header=element('tr');
  for(const label of [t.quantity,t.value,t.unit]){const th=element('th',label);th.scope='col';header.append(th);}head.append(header);table.append(head);
  const row=(name,value,unit)=>{const tr=element('tr');tr.append(element('th',name),element('td',value),element('td',unit));body.append(tr);};
  if(r.schema==='societybox-input-output-result-v1'){
    for(let i=0;i<r.sector_names.length;i++)row(r.sector_names[i],r.gross_output_change_exact[i],r.unit);
  }else if(r.schema==='societybox-survey-replicate-result-v1'){
    row(t.mean,r.point_exact,r.unit);row(t.variance,r.variance_exact,r.unit+' squared');
  }else if(r.schema==='societybox-vintage-comparison-result-v1'){
    for(const change of r.changes){
      const value=t.vintage_before+': '+(change.earlier_value_decimal??'—')+' → '+t.vintage_after+': '+(change.later_value_decimal??'—')+
        '; '+t.vintage_delta+': '+(change.delta_exact??(change.status==='UNRESOLVED_AVAILABILITY'?t.vintage_unresolved:'—'));
      row(change.series_id+' · '+change.period_start+' · '+t['vintage_status_'+change.status]+' ['+change.status+']',value,change.unit);
    }
  }else if(r.coefficients_exact&&r.parameter_order){
    for(let i=0;i<r.parameter_order.length;i++)row(r.parameter_order[i],r.coefficients_exact[i],r.parameter_units[i]);
  }else if(r.horizon_summaries){
    for(const h of r.horizon_summaries)if(h.scored_pairs){row('h'+h.horizon+' · '+t.model_loss,h.model_mse_exact,details.chart.unit);row('h'+h.horizon+' · '+t.baseline_loss,h.baseline_mse_exact,details.chart.unit);}
  }else if(details.chart){
    const values=details.chart.values_exact||details.chart.values_decimal||[];
    for(let i=0;i<values.length;i++)row(details.chart.labels[i],values[i],details.chart.unit);
  }
  table.append(body);$('result-values').append(table);
  $('report').srcdoc=details.html;$('report').hidden=false;
  for(const row of details.evidence.items){const article=element('article');article.append(element('h3',row.kind),element('pre',JSON.stringify(row.record,null,2)));$('inspector-records').append(article);}
  text($('evidence-count'),details.evidence.offset+' / '+details.evidence.total);$('evidence-count').dataset.total=details.evidence.total;
  controls();
}
$('composer').onsubmit=event=>{event.preventDefault();work(async()=>{
  const next=await api('/api/message',{text:$('question').value,expected_revision:state.revision});
  messageOffset=null;render(next);$('question').value='';
});};
$('model-form').onsubmit=event=>{event.preventDefault();work(async()=>{
  const spec=JSON.parse($('spec').value);
  const question=$('question').value.trim()||state.latest_question;if(question)spec.question=question;
  const next=await api('/api/proposal',{spec,expected_revision:state.revision});
  selected=null;dirty=false;versionOffset=null;render(next);$('question').value='';
});};
$('spec').addEventListener('input',changed);$('question').addEventListener('input',changed);
$('approve').onchange=controls;
$('execute').onclick=()=>work(async()=>{
  if(dirty||!selected||!$('approve').checked)throw new Error('REVIEW_REQUIRED');
  const job=await api('/api/jobs',{version_id:selected.id,acknowledgment:selected.review_sha256,expected_revision:state.revision,confirmed:true});
  $('approve').checked=false;activeJob=job;displayJob(job);schedulePoll();
});
function displayJob(job){
  viewedJob=job;
  const label=job.version_id+' · '+(t['job_'+job.status]||job.status)+' · '+job.id.slice(0,8);
  if($('job-status').textContent!==label)text($('job-status'),label);
  $('job-status').dataset.state=job.status;controls();
}
async function refreshJobs(){
  const page=await api('/api/jobs?offset='+jobOffset+'&size=40');jobTotal=page.total;
  text($('job-history-count'),jobOffset+' / '+jobTotal);$('job-history').replaceChildren();
  for(const job of page.items){
    const button=element('button',job.version_id+' · '+(t['job_'+job.status]||job.status));button.type='button';
    button.onclick=()=>{if(job.run_id)work(()=>showRun(job.run_id));else if(!activeJob)displayJob(job);};$('job-history').append(button);
  }
  if(!activeJob){const running=page.items.find(job=>!terminal(job));if(running){activeJob=running;displayJob(running);schedulePoll();}}
  const recovery=page.items.find(job=>job.status==='RECOVERY_REQUIRED');
  if(recovery){capabilities.recovery_required=true;if(!activeJob)displayJob(recovery);}
  controls();
}
function schedulePoll(delay){if(pollTimer!==null)clearTimeout(pollTimer);pollTimer=setTimeout(pollJob,delay??(activeJob?.status==='CANCEL_REQUESTED'?250:1000));}
async function pollJob(){
  pollTimer=null;if(!activeJob)return;
  try{
    const key=activeJob.id,job=await api('/api/jobs/'+key+'?wait=1');
    if(activeJob?.id!==key)return;
    activeJob=job;displayJob(job);
    if(!terminal(job)){schedulePoll();return;}
    activeJob=null;controls();await refresh();await refreshJobs();
    if(job.status==='SUCCEEDED'&&selected?.id===job.version_id&&!dirty){evidenceOffset=0;await showRun(job.run_id);}
    if(job.status!=='SUCCEEDED')status(t['job_'+job.status]||t.error,job.status!=='CANCELLED');
  }catch(error){status(activeJob?t.job_unavailable:t.error,true);if(activeJob&&error.status!==401&&error.status!==410)schedulePoll(5000);}
}
$('cancel-job').onclick=async()=>{
  if(!activeJob)return;
  const key=activeJob.id;
  try{const job=await api('/api/jobs/'+key+'/cancel',{});if(activeJob?.id!==key)return;activeJob=job;displayJob(job);schedulePoll();}
  catch(error){status(t.error,true);}
};
$('recover-job').onclick=()=>work(async()=>{
  if(!viewedJob)return;
  const job=await api('/api/jobs/'+viewedJob.id+'/recover',{acknowledgment:viewedJob.id});displayJob(job);
  capabilities=await api('/api/capabilities?locale='+locale);await refreshJobs();
});
for(const [id,delta] of [['job-history-prev',-40],['job-history-next',40]])$(id).onclick=()=>work(async()=>{jobOffset=Math.max(0,jobOffset+delta);await refreshJobs();});
$('family').onchange=()=>{if(templates[$('family').value]){$('spec').value=JSON.stringify(templates[$('family').value],null,2);changed();}};
$('study-file').onchange=()=>work(async()=>{
  const file=$('study-file').files[0];if(!file)return;if(file.size>65536)throw new Error('SIZE');
  const raw=await file.text(),spec=JSON.parse(raw);$('spec').value=raw;
  if(templates[spec.family])$('family').value=spec.family;
  changed();$('study-file').value='';
});
for(const [id,delta] of [['messages-prev',-40],['messages-next',40]])$(id).onclick=()=>work(async()=>{messageOffset=Math.max(0,state.message_offset+delta);await refresh();});
for(const key of ['versions','runs'])for(const [direction,delta] of [['prev',-40],['next',40]])$(key+'-'+direction).onclick=()=>work(async()=>{
  const offset=Math.max(0,state[key.slice(0,-1)+'_offset']+delta);if(key==='versions')versionOffset=offset;else runOffset=offset;await refresh();
});
for(const [id,delta] of [['evidence-prev',-30],['evidence-next',30]])$(id).onclick=()=>work(async()=>{const id=selectedRun;evidenceOffset=Math.max(0,evidenceOffset+delta);await showRun(id);});
$('download').onclick=()=>work(async()=>{
  const response=await fetch('/api/download/'+selectedRun+'/bundle.zip',{headers:{Authorization:'Bearer '+token},credentials:'omit',cache:'no-store',redirect:'error'});
  if(!response.ok)throw new Error('DOWNLOAD_REJECTED');
  const url=URL.createObjectURL(await response.blob()),link=element('a');link.href=url;link.download='societybox-'+selectedRun+'.zip';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function connect(){
  [templates,capabilities,localModel]=await Promise.all([api('/api/templates'),api('/api/capabilities?locale='+locale),api('/api/conversation/status')]);
  const configured=localModel.mode==='LOCAL_LLM_CONFIGURED_NOT_CONNECTED';
  $('local-llm').hidden=!configured;
  text($('llm-mode'),configured ? ('LOCAL_LLM_CONFIGURED_NOT_CONNECTED · '+localModel.local_provider+' · '+t.llm_disclosure) : t.no_llm);
  $('family').replaceChildren();
  for(const policy of capabilities.families){const option=element('option',policy.name+(policy.blockers.length?' · '+t.blocked:''));option.value=policy.family;option.title=policy.blockers.join(', ');$('family').append(option);}
  text($('engine-configuration'),t.runtime+': '+capabilities.runtime+' · '+t.archives+': '+capabilities.sources);
  if(!selected)$('spec').value=JSON.stringify(templates.input_output,null,2);
  await refresh();
  await refreshJobs();
  if(localModel.mode==='LOCAL_LLM_CONFIGURED_NOT_CONNECTED')await refreshConversationTurns();
}
$('connect').onclick=()=>work(async()=>{token=$('access-key').value.trim();$('access-key').value='';await connect();});
window.addEventListener('pageshow',event=>{if(event.persisted){dirty=true;$('approve').checked=false;clearResult();if(token)work(refresh);}});
document.querySelectorAll('a[data-locale]').forEach(link=>link.onclick=event=>{
  event.preventDefault();const url=new URL(link.getAttribute('href'),location.origin);url.hash=token;location.assign(url.href);
});
async function refreshConversationTurns(){
  const page=await api('/api/conversation/turns?offset=0&size=20');
  $('llm-history').replaceChildren();
  for(const row of page.items){
    const button=element('button',row.id.slice(0,8)+' · '+row.status);
    button.type='button';
    button.onclick=()=>work(async()=>{const turn=await api('/api/conversation/turn/'+row.id);showLocalTurn(turn);});
    $('llm-history').append(button);
  }
}
function showLocalTurn(turn){
  localReview=turn.review;
  localTurn={id:turn.id,status:turn.status};
  $('llm-selected-text').value=turn.review.text;
  $('llm-classification').value=turn.review.classification;
  $('llm-confirm').checked=false;
  text($('llm-review-summary'),turn.review.scope+' · '+turn.review.review_sha256+'\n'+turn.review.text);
  $('llm-candidates').replaceChildren();
  if(turn.result){
    text($('llm-response'),turn.result.assistant_text+'\n'+turn.result.claim_verification+' · '+turn.result.model_instruction_consumption);
    for(const candidate of turn.result.tool_calls){
      if(candidate.name!=='propose_engine_study'||candidate.authorization!=='NOT_GRANTED')continue;
      const button=element('button',t.llm_load_draft+' · '+candidate.id);button.type='button';
      button.onclick=()=>work(async()=>{
        const draft=await api('/api/conversation/draft',{turn_id:turn.id,call_id:candidate.id});
        $('spec').value=JSON.stringify(draft,null,2);$('family').value=draft.family;
        changed();text($('llm-response'),turn.result.assistant_text+'\n'+t.llm_draft_warning);
      });
      $('llm-candidates').append(button);
    }
  }else text($('llm-response'),turn.status+' · '+(turn.error||''));
  controls();
}
$('llm-selected-text').oninput=()=>{
  localReview=null;localTurn=null;$('llm-confirm').checked=false;
  text($('llm-review-summary'),t.llm_review_required);$('llm-candidates').replaceChildren();
  controls();
};
$('llm-classification').onchange=()=>$('llm-selected-text').oninput();
$('llm-review-button').onclick=()=>work(async()=>{
  const selected=$('llm-selected-text').value;
  const checked=await api('/api/conversation/review',{
    text:selected,locale,expected_revision:state.revision,classification:$('llm-classification').value});
  localReview=checked;localTurn=null;$('llm-confirm').checked=false;
  text($('llm-review-summary'),checked.scope+' · '+checked.review_sha256+'\n'+checked.text);
  text($('llm-response'),t.llm_review_required);
});
$('llm-confirm').onchange=controls;
$('llm-begin').onclick=()=>work(async()=>{
  if(!localReview||!$('llm-confirm').checked)throw new Error('LLM_REVIEW_REQUIRED');
  localTurn=await api('/api/conversation/begin',{
    text:localReview.text,locale:localReview.locale,expected_revision:localReview.expected_revision,
    classification:localReview.classification,acknowledgment:localReview.review_sha256,confirmed:true});
  $('llm-confirm').checked=false;text($('llm-response'),t.llm_granted_not_sent);
  await refreshConversationTurns();
});
$('llm-send').onclick=async()=>{
  if(busy||localSending||!localTurn||!localReview)return;
  localSending=true;
  const key=localTurn.id;localTurn.status='IN_FLIGHT';controls();
  text($('llm-response'),'IN_FLIGHT · '+t.llm_disclosure);
  try{
    await api('/api/conversation/send',{turn_id:key,acknowledgment:localReview.review_sha256,confirmed:true});
  }catch(error){status(t.error+' · '+(error.status||''),true);
  }finally{
    try{
      const turn=await api('/api/conversation/turn/'+key);showLocalTurn(turn);await refreshConversationTurns();
    }catch(error){status(t.error+' · '+(error.status||''),true);
    }finally{localSending=false;controls();}
  }
};
controls();if(token)work(connect);else status(t.locked);
})();
"""


def render(locale='ko'):
    from .research_review_entry import _hash
    from .feature_catalog import describe_feature
    if locale not in _TEXT: raise ValueError('CHAT_LOCALE_INVALID')
    labels=dict(zip(_KEYS,_TEXT[locale])); s=lambda key:escape(labels[key])
    labels.update(zip(('runtime','archives','blocked','quantity','value','unit','mean','variance','model_loss','baseline_loss'),{
        'en':('Pinned runtime','Pinned sources','Blocked','Quantity','Value','Unit','Mean','Variance','Model MSE','Baseline MSE'),
        'ko':('고정 런타임','고정 원본','차단됨','변수·통계량','값','단위','평균','분산','모형 MSE','기준선 MSE'),
        'zh-Hans':('固定运行时','固定来源','已阻止','变量·统计量','值','单位','均值','方差','模型MSE','基准MSE'),
        'ja':('固定ランタイム','固定ソース','ブロック中','変数・統計量','値','単位','平均','分散','モデルMSE','基準MSE'),
    }[locale]))
    labels.update(zip(('vintage_before','vintage_after','vintage_delta','vintage_unresolved'),{
        'en':('Earlier cutoff','Later cutoff','Exact revision difference','Unavailable or unresolved; no zero/latest-value fallback'),
        'ko':('이전 기준시점','이후 기준시점','정확한 개정 차이','이용 불가 또는 시점 불명확: 0·최신 값으로 대체하지 않음'),
        'zh-Hans':('较早截止时点','较晚截止时点','精确修订差异','不可用或时点不明；不替换为零或最新值'),
        'ja':('前の基準時点','後の基準時点','正確な改定差分','利用不可または時点不明。ゼロ・最新値で補完しません'),
    }[locale]))
    labels.update(zip(('vintage_status_COMPARABLE','vintage_status_ADDED','vintage_status_REMOVED','vintage_status_NOT_AVAILABLE','vintage_status_UNRESOLVED_AVAILABILITY'),{
        'en':('Comparable','Newly available','No longer selected','Unavailable at both cutoffs','Availability unresolved'),
        'ko':('비교 가능','새로 이용 가능','이후 선택에서 제외','두 시점 모두 이용 불가','이용시점 불명확'),
        'zh-Hans':('可比较','新增可用值','不再被选中','两个时点均不可用','可用时点不明'),
        'ja':('比較可能','新たに利用可能','後の選択から除外','両時点で利用不可','利用時点不明'),
    }[locale]))
    labels.update(zip(('recover_job','recovery_help'),{
        'en':('Review recovery','Calculation is blocked until the previous worker is independently checked. Only a stopped job observed in this session can be recovered here. You can continue in a new empty workspace.'),
        'ko':('복구 검토','이전 worker를 독립적으로 확인하기 전에는 계산이 차단됩니다. 현재 세션에서 종료가 관측된 작업만 여기서 복구할 수 있습니다. 새 빈 workspace에서 작업을 계속할 수 있습니다.'),
        'zh-Hans':('恢复审查','须独立检查之前的worker，才能继续计算。这里只能恢复本会话中已观测到停止的任务。可在新的空工作空间继续。'),
        'ja':('復旧確認','以前のworkerを独立に確認するまで計算は停止されます。このセッションで停止を観測したジョブのみ復旧できます。新しい空のworkspaceで続行できます。'),
    }[locale]))
    labels.update(zip(('jobs','cancel_job','job_PENDING','job_RUNNING','job_CANCEL_REQUESTED','job_SUCCEEDED','job_CANCELLED','job_FAILED','job_EXPIRED','job_RECOVERY_REQUIRED','job_unavailable'),{
        'en':('Study jobs','Cancel calculation','Pending','Running','Cancellation requested','Completed','Cancelled','Failed','Expired','Recovery review required','Job status unavailable; execution is not assumed stopped.'),
        'ko':('연구 작업','계산 취소','대기 중','계산 중','취소 요청됨','계산 완료','취소됨','실패','시간 만료','복구 검토 필요','작업 상태를 확인할 수 없습니다. 중지되었다고 간주하지 않습니다.'),
        'zh-Hans':('研究任务','取消计算','等待中','计算中','已请求取消','已完成','已取消','失败','已超时','需要恢复审查','无法读取任务状态；不假定任务已停止。'),
        'ja':('研究ジョブ','計算を中止','待機中','計算中','中止要求済み','完了','中止済み','失敗','期限切れ','復旧確認が必要','ジョブ状態を取得できません。停止したとは判断しません。'),
    }[locale]))
    labels.update(zip((
        'llm_title','llm_selected','llm_class','llm_public','llm_private',
        'llm_review','llm_confirm','llm_grant','llm_send','llm_response',
        'llm_history','llm_load_draft','llm_disclosure','llm_review_required',
        'llm_granted_not_sent','llm_draft_warning'),{
        'en':('Optional local model','Explicit selected text to local model','Data classification',
              'Public or synthetic','Private text, explicitly approved for local-only transfer',
              'Review selected text (no transfer)','I authorize only the displayed selected text to my configured local model.',
              'Record transfer authorization (no transfer)','Send once to local model',
              'Untrusted model response','Previous model turns','Load proposal into editable study',
              'No history or research JSON is automatically sent. Local model output has no tools or scientific authority.',
              'Review the exact selected text before approving any local transfer.',
              'Authorization recorded. No text has been sent. Use the separate Send action.',
              'Untrusted editable draft only. Saving and executing the model requires separate review and approval.'),
        'ko':('선택형 로컬 모델','로컬 모델에 보낼 명시적 선택 텍스트','자료 분류',
              '공개 또는 합성','비공개 텍스트: 로컬 전송을 별도 승인',
              '선택 텍스트 검토 (전송 없음)','표시된 텍스트만 설정한 로컬 모델에 보내는 것을 승인합니다.',
              '전송 승인 기록 (아직 전송 안 함)','로컬 모델에 한 번 전송',
              '비신뢰 모델 응답','이전 모델 대화','제안을 편집 가능한 연구 초안으로 열기',
              '대화 이력·연구 JSON은 자동 전송하지 않습니다. 모델 출력에 계산 권한이나 학술적 권위가 없습니다.',
              '선택한 전송 텍스트를 정확히 검토한 후 승인하세요.',
              '승인은 기록되었고 아직 전송되지 않았습니다. 별도 전송 버튼을 사용하세요.',
              '검증되지 않은 편집 초안입니다. 저장과 계산에는 각각 별도 검토·승인이 필요합니다.'),
        'zh-Hans':('可选本地模型','明确选择要发送给本地模型的文本','数据分类',
              '公开或合成','私有文本：单独批准本地传输',
              '审查选定文本（不发送）','仅授权将显示的文本发送给配置的本地模型。',
              '记录传输批准（尚未发送）','向本地模型发送一次',
              '不可信模型响应','先前的模型轮次','加载为可编辑的研究草稿',
              '不会自动发送历史消息或研究JSON。模型输出没有工具执行权或科学证明效力。',
              '先审查确切文本，再批准本地传输。',
              '批准已记录，尚未发送。请使用独立发送按钮。',
              '仅不可信的可编辑草稿；保存和计算需要分别审查批准。'),
        'ja':('任意のローカルモデル','ローカルモデルへ明示的に送るテキスト','データ区分',
              '公開・合成','非公開文：ローカル送信を個別承認',
              '選択テキストを確認（送信なし）','表示したテキストだけを設定したローカルモデルに送ることを承認します。',
              '送信承認を記録（送信はまだ）','ローカルモデルに一度送信',
              '信頼されないモデル応答','過去のモデル対話','提案を編集可能な研究下書きへ',
              '履歴や研究JSONは自動送信しません。モデル出力に計算権限や学術的証明力はありません。',
              '実際に送るテキストを確認してから承認してください。',
              '承認を記録しましたが未送信です。別の送信ボタンを使用してください。',
              '未検証の編集用下書きです。保存と計算には別途確認と承認が必要です。'),
    }[locale]))
    def pager(name):
        return ('<div class="pager"><button type="button" id="'+name+'-prev">'+s('previous')+'</button>'
                '<span id="'+name+'-count"></span><button type="button" id="'+name+'-next">'+s('next')+'</button></div>')
    options=''.join('<option value="'+family+'">'+escape(describe_feature(family,locale)['name'])+'</option>'
                    for family in ('input_output','survey_replicate'))
    csp="default-src 'none'; script-src 'sha256-"+_hash(SCRIPT)+"'; style-src 'sha256-"+_hash(STYLE)+"'; connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'"
    return ('<!doctype html><html lang="'+locale+'"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="'+escape(csp,quote=True)+'"><title>Society Box · '+s('title')+'</title><style>'+STYLE+'</style></head><body>'
        '<header><div class="bar"><h1>Society Box · '+s('title')+'</h1><div>'
        +''.join('<a data-locale="'+code+'" href="/?locale='+code+'">'+code+'</a> ' for code in _TEXT)
        +'</div></div><p class="muted" id="llm-mode">'+s('no_llm')+'</p><p id="engine-configuration" class="muted"></p><p id="status" role="status" aria-live="polite"></p></header>'
        '<section id="access" class="card"><p>'+s('locked')+'</p><label for="access-key">'+s('key')+'</label><input id="access-key" type="password" autocomplete="off">'
        '<button id="connect" type="button">'+s('connect')+'</button></section>'
        '<div id="workspace" class="layout"><nav aria-label="'+s('navigation')+'"><details open><summary>'+s('navigation')+'</summary><h2>'+s('versions')+'</h2>'
        '<div id="versions"></div>'+pager('versions')+'<h2>'+s('runs')+'</h2><div id="runs"></div>'+pager('runs')
        +'<h2>'+s('jobs')+'</h2><div id="job-history"></div>'+pager('job-history')+'</details></nav>'
        '<main><h2>'+s('conversation')+'</h2>'+pager('messages')+'<div id="messages" role="log" aria-label="'+s('conversation')+'"></div>'
        '<form id="composer"><label for="question">'+s('question')+'</label><textarea id="question" rows="3" maxlength="2000"></textarea><button class="primary" type="submit">'+s('send')+'</button></form>'
        '<section id="local-llm" class="card" hidden><h2>'+s('llm_title')+'</h2><p>'+s('llm_disclosure')+'</p>'
        '<label for="llm-selected-text">'+s('llm_selected')+'</label><textarea id="llm-selected-text" rows="3" maxlength="4000"></textarea>'
        '<label for="llm-classification">'+s('llm_class')+'</label><select id="llm-classification">'
        '<option value="PUBLIC_OR_SYNTHETIC">'+s('llm_public')+'</option>'
        '<option value="PRIVATE_APPROVED_LOCAL">'+s('llm_private')+'</option></select>'
        '<button id="llm-review-button" type="button">'+s('llm_review')+'</button>'
        '<pre id="llm-review-summary">'+s('llm_review_required')+'</pre>'
        '<label class="check"><input id="llm-confirm" type="checkbox" disabled><span>'+s('llm_confirm')+'</span></label>'
        '<button id="llm-begin" type="button" disabled>'+s('llm_grant')+'</button>'
        '<button id="llm-send" type="button" disabled>'+s('llm_send')+'</button>'
        '<h3>'+s('llm_response')+'</h3><pre id="llm-response" role="status" aria-live="polite"></pre>'
        '<div id="llm-candidates"></div><details><summary>'+s('llm_history')+'</summary><div id="llm-history"></div></details></section>'
        '<form id="model-form"><h2>'+s('model')+'</h2><p class="muted">'+s('examples')+'</p><label for="family">'+s('model')+'</label><select id="family">'+options+'</select>'
        '<label for="study-file">'+s('file')+'</label><input class="json-file" type="file" id="study-file" accept=".json,application/json">'
        '<details open><summary>'+s('edit')+'</summary><label for="spec">'+s('model')+'</label><textarea id="spec" rows="12" spellcheck="false"></textarea></details>'
        '<button id="review-model" type="submit">'+s('review')+'</button></form><section class="card"><h2>'+s('approval')+'</h2><p id="review-digest"></p>'
        '<label class="check"><input id="approve" type="checkbox" disabled><span>'+s('confirm')+'</span></label><button class="primary" id="execute" type="button" disabled>'+s('run')+'</button>'
        '<p id="job-status" role="status" aria-live="polite"></p><button id="cancel-job" type="button" disabled>'+s('cancel_job')+'</button>'
        '<p id="recovery-note" hidden>'+s('recovery_help')+'</p><button id="recover-job" type="button" disabled>'+s('recover_job')+'</button></section>'
        '<section class="card"><h2>'+s('result')+'</h2><p id="execution-status">'+s('no_result')+'</p><div id="result-values" class="table-scroll"></div>'
        '<details><summary>'+s('result')+' · NO_LLM</summary><pre id="result-summary">'+s('no_result')+'</pre></details><button id="download" type="button" disabled>'+s('export')+'</button>'
        '<iframe id="report" title="'+s('report')+'" sandbox="" hidden></iframe></section></main>'
        '<aside id="inspector" aria-label="'+s('inspector')+'"><details open><summary>'+s('inspector')+'</summary><h2>'+s('source')+'</h2><p>'+s('rights')+'</p><pre id="source-info"></pre>'
        '<p>'+s('unknown')+'</p>'+pager('evidence')+'<div id="inspector-records"></div></details></aside></div>'
        '<script id="labels" type="application/json">'+json.dumps(labels,ensure_ascii=False).replace('<','\\u003c')+'</script><script>'+SCRIPT+'</script></body></html>')

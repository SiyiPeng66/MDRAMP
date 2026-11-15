"""Create portable production configuration templates without touching datasets."""
import argparse
import json
from pathlib import Path
import yaml


def templates(workspace):
    prepared='data/prepared'; output='outputs/production'
    def stage(name,module,options,inputs,outputs):
        arguments=[]
        for flag,value in options.items():
            arguments.append('--'+flag)
            arguments.append({'path':value} if value in inputs+outputs else value)
        return dict(name=name,module=module,args=arguments,inputs=inputs,outputs=outputs)
    pos=f'{prepared}/public_features.csv'; bg=f'{prepared}/background_features.csv'
    pe=f'{prepared}/public.npz'; pm=f'{prepared}/public_metadata.csv'
    be=f'{prepared}/background.npz'; bm=f'{prepared}/background_metadata.csv'
    teacher=f'{prepared}/pore_scores.csv'; scaler=f'{output}/M_R0/feature_scaler.json'
    checkpoint=f'{output}/M_R0/experts'
    steps=[stage('train_experts','src.train.pretrain_public',{
      'positive-features':pos,'positive-embeddings':pe,'positive-embedding-metadata':pm,
      'background-features':bg,'background-embeddings':be,'background-embedding-metadata':bm,
      'pore-scores':teacher,'id-column':'peptide_id','checkpoint-dir':checkpoint,
      'scaler-output':scaler,'epochs':200,'batch-size':64,'learning-rate':.0001,'device':'cpu'},
      [pos,pe,pm,bg,be,bm,teacher],[checkpoint,scaler])]
    pair_features=f'{prepared}/pair_features.csv'; pair_emb=f'{prepared}/pair_peptides.npz'; pair_meta=f'{prepared}/pair_metadata.csv'
    targets=f'{prepared}/targets.npz'; target_meta=f'{prepared}/target_metadata.csv'; pairs=f'{prepared}/interaction_pairs.csv'
    interaction=f'{output}/M_R0/interaction.pt'
    steps.append(stage('train_interaction','src.train.train_target_affinity',{
      'pair-scores':pairs,'peptide-features':pair_features,'peptide-embeddings':pair_emb,'peptide-embedding-metadata':pair_meta,
      'target-embeddings':targets,'target-embedding-metadata':target_meta,'peptide-id-column':'peptide_id','scaler':scaler,
      'curation-manifest':f'{prepared}/interaction_curation_manifest.json','data-mode':'strict-v6',
      'output':interaction,'epochs':200,'batch-size':32,'learning-rate':.00005,'device':'cpu'},
      [pairs,pair_features,pair_emb,pair_meta,targets,target_meta,scaler,f'{prepared}/interaction_curation_manifest.json'],[interaction]))
    initial=f'{prepared}/initial_peptides.csv'; initial_features=f'{prepared}/initial_features.csv'; initial_emb=f'{prepared}/initial.npz'; initial_meta=f'{prepared}/initial_metadata.csv'
    pair_predictions=f'{output}/M_R0/init_pairs.csv'; coverage=f'{output}/M_R0/init_coverage.csv'
    steps.append(stage('initial_interaction_features','src.score.predict_target_affinity',{
      'checkpoint':interaction,'scaler':scaler,'peptide-features':initial_features,'peptide-embeddings':initial_emb,
      'peptide-embedding-metadata':initial_meta,'target-embeddings':targets,'target-embedding-metadata':target_meta,
      'peptide-id-column':'peptide_id','pair-output':pair_predictions,'aggregate-output':coverage,'device':'cpu'},
      [interaction,scaler,initial_features,initial_emb,initial_meta,targets,target_meta],[pair_predictions,coverage]))
    means=f'{output}/M_R0/initial_expert_means.csv'; derived=f'{output}/M_R0/initial_features.csv'
    prior=f'{checkpoint}/amp_prior_ensemble.pt'; membrane=f'{checkpoint}/membrane_ensemble.pt'
    item=stage('initial_expert_means','src.score.score_candidates',{
      'candidates':initial,'features':derived,'embeddings':initial_emb,'embedding-metadata':initial_meta,
      'id-column':'peptide_id','scaler':scaler,'amp-prior':prior,'membrane':membrane,'target-affinity':coverage,'output':means},
      [initial,initial_emb,initial_meta,scaler,prior,membrane,coverage],[derived,means])
    item['args'].append('--experts-only');steps.append(item)
    ledger=f'{prepared}/initial_mic_ledger.csv'; manifest=f'{prepared}/state_R0.json'; mic=f'{output}/M_R0/mic.pt'
    steps.append(stage('initialize_mic','src.train.train_mic_head',{
      'mic-ledger':ledger,'peptide-features':derived,'peptide-embeddings':initial_emb,'peptide-embedding-metadata':initial_meta,
      'expert-means':means,'state-manifest':manifest,'current-round':'R0','scaler':scaler,'peptide-id-column':'peptide_id',
      'output':mic,'epochs':200,'batch-size':32,'learning-rate':.0001,'weight-decay':.0001,'patience':15,'seed':11},
      [ledger,derived,initial_emb,initial_meta,means,manifest,scaler],[mic]))
    generator_data=f'{prepared}/public_sequences.csv'; generator=f'{output}/generator/reference.pt'
    item=stage('pretrain_generator','src.generation.cli',{'data':generator_data,'output':generator,'seed':11,'epochs':100,'batch-size':128,'learning-rate':.0001,'weight-decay':.0001,'device':'cpu'},[generator_data],[generator])
    item['args'].insert(0,'pretrain');steps.append(item)
    bootstrap={'schema_version':1,'workspace':workspace,'steps':steps}
    # Each CLI command config is relative to its own file, unlike workflow workspace.
    scorer={'path_base':workspace,'esm_checkpoint':'REPLACE_WITH_REGISTERED_ESM_SNAPSHOT/data/weights/esm3_sm_open_v1.pth',
      'scaler':scaler,'amp_prior':prior,'membrane':membrane,'target_affinity':interaction,'mic_checkpoint':mic,
      'target_embeddings':targets,'target_embedding_metadata':target_meta,'state_manifest':manifest,'panel_manifest':f'{prepared}/panel_manifest.json','device':'cpu','pair_batch_size':1024,
      'aggregation_coefficient':.8,'kappa':1.,'tau':.5,'gamma':10.,'lambda_syn':.3}
    predict={'schema_version':1,'command':'predict','options':{
       'input':f'{workspace}/data/candidates.fasta','scorer-config':'scorer_R0.json',
       'output':f'{workspace}/{output}/predictions/candidates.csv','batch-size':128,'device':'cpu'}}
    mic_config={'schema_version':1,'command':'train mic','options':{
       'mic-ledger':f'{workspace}/{prepared}/cumulative_R0.csv','peptide-features':f'{workspace}/{prepared}/cumulative_features.csv',
       'peptide-embeddings':f'{workspace}/{prepared}/cumulative.npz','peptide-embedding-metadata':f'{workspace}/{prepared}/cumulative_metadata.csv',
       'expert-means':f'{workspace}/{prepared}/cumulative_expert_means.csv','state-manifest':f'{workspace}/{prepared}/state_R1.json',
       'current-round':'R1','parent-checkpoint':f'{workspace}/{mic}','scaler':f'{workspace}/{scaler}',
       'peptide-id-column':'peptide_id','output':f'{workspace}/{output}/M_R1/mic.pt','epochs':100,'batch-size':32,
       'learning-rate':.00005,'weight-decay':.0001,'patience':15,'seed':11,'device':'cpu'}}
    # These RL values are explicit engineering choices, not manuscript-reported settings.
    policy={'schema_version':1,'command':'train policy','options':{
      'checkpoint':f'{workspace}/{generator}','reference':f'{workspace}/{generator}',
      'scorer-config':'scorer_R0.json','reference-sequences':f'{workspace}/{prepared}/generation_exclusion_references.csv',
      'output':f'{workspace}/{output}/generator/policy_R0.pt','seed':11,'device':'cpu',
      'steps':100,'beta-kl':.1,'learning-rate':.00001,'batch-size':32,'weight-decay':.0001}}
    sample={'schema_version':1,'command':'generate','options':{
      'checkpoint':f'{workspace}/{output}/generator/policy_R0.pt','output':f'{workspace}/{output}/generator/R0_proposals.csv',
      'seed':11,'count':200,'temperature':1.,'top-p':.95,'device':'cpu',
      'reference-sequences':f'{workspace}/{prepared}/generation_exclusion_references.csv',
      'eligible-output':f'{workspace}/{output}/generator/R0_eligible.csv'}}
    check={'schema_version':1,'command':'generation check','options':{'scorer-config':'scorer_R0.json'}}
    return {'policy_R0.yaml':yaml.safe_dump(policy,sort_keys=False),
            'generate_R0.yaml':yaml.safe_dump(sample,sort_keys=False),
            'check_generation.yaml':yaml.safe_dump(check,sort_keys=False),
            'bootstrap.yaml':yaml.safe_dump(bootstrap,sort_keys=False), 'predict.yaml':yaml.safe_dump(predict,sort_keys=False),
            'update_R1.yaml':yaml.safe_dump(mic_config,sort_keys=False),'scorer_R0.json':json.dumps(scorer,indent=2)+'\n'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--workspace',default='.',help='Workspace path relative to new config directory')
    args=parser.parse_args(); directory=Path(args.directory)
    files=templates(args.workspace)
    if any((directory/name).exists() for name in files): parser.error('Refusing to overwrite existing configuration')
    directory.mkdir(parents=True,exist_ok=True)
    for name,contents in files.items(): (directory/name).write_text(contents)
    print(f'Created {len(files)} production configurations in {directory}; fill prepared inputs and registered ESM checkpoint before running')
if __name__=='__main__':main()

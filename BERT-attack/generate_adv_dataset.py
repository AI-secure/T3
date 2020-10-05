import json

def generate_adv_dataset(input_file, adv_dic):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    output_data={'data':[],'version':input_data['version']}
    dataset=input_data["data"]
    for article in dataset:
        new_article=article.copy()
        new_article['paragraphs']=[]
        
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                new_context=paragraph['context']+adv_dic.get(qa['id'],"")
                new_pqa={'context':new_context,'qas':[qa.copy()]}
                new_article['paragraphs'].append(new_pqa)
        output_data['data'].append(new_article)
        
    with open('adv_{}'.format(input_file),"w") as f:
        json.dump(output_data,f)
        
generate_adv_dataset("dev-v1.1.json",{})       
    
    
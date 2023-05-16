from lfqa import get_csv

all_docs_dir = "/content/drive/MyDrive/GPL/Petal_physiology/"
queries=[
"biological systems from this reading are in aqueous solutions, how would these strategies apply for other solvents?",
"what solutes can be excreted in urine?",
"how does fluid move through the kidneys?",
"how does is plasma filtered into primary urine?",
"what structures compose the filter between plasma and primary urine?",
"what structures are found in the kidney?",
"compare and contrast blood plasma and urine"
]

def export_csv(queries, all_docs_dir):
    df = get_csv(queries, all_docs_dir)
    df.to_excel("Results_Physiology_testing_2.xlsx", index=None)

if __name__=='__main__':
    export_csv(queries, all_docs_dir)
    
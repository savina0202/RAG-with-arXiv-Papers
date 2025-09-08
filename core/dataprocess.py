import arxiv
import os
import json
import config


def data_collection(max_results=50, categories="cs.CL", save_paperlist=True, paperlist_filename=config.PAPERLIST_FILENAME):

    query = f"cat:{categories}"

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results = search.results()

    if not results:
        print("No results found.")  
        return None
    else:    
        #print(f"Found {len(list(results))} results.")

        # Create a dir to save PDFs if it doesn't exist
        os.makedirs(config.PAPER_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)

        paperlist = []
        print(f"Starting download of {max_results} papers in category {categories}...")
        for index, result in enumerate(results):
            text = f"{index+1}. Title: {result.title} downloading ..."
            doc_id=result.entry_id.split('/')[-1]
            filename=result.entry_id.split('/')[-1]+'.pdf'
            result.download_pdf(dirpath=config.PAPER_DIR, filename=filename)
            paperlist.append({
                "doc_id": doc_id,
                "paper_title": result.title,
                "filename": filename,
                "summary": result.summary[:200],
                "url": result.pdf_url,
                "authors": ', '.join([author.name for author in result.authors]),
                "publish_date": f"{result.published.year}-{str(result.published.month).zfill(2)}-{str(result.published.day).zfill(2)}"
            })
            print(f"{text} ... Done")
            index += 1

        
        if save_paperlist:
            with open(paperlist_filename, 'w', encoding="utf-8") as f:
                json.dump(paperlist, f, ensure_ascii=False, indent=4)
            print(f"Paper list saved to {paperlist_filename}")

        print("Download completed!")
        return(paperlist)
    
if __name__ == "__main__":
    data_collection(50, "cs.CL", True, "paperlist.json")

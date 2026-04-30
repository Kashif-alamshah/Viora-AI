from langchain_community.document_loaders import PubMedLoader,ArxivLoader
from langchain.tools import tool
from predictor import predict
from cnn_predictor import cnn_predict
from oral_predictor import oral_predict

@tool
def oral_cancer_predictor(image_path: str) -> str:
    """
    Analyzes an oral lesion image using a CNN model.
    Predicts whether the lesion is Benign or Malignant.
    Also generates a GradCAM heatmap showing which regions influenced the decision.

    Args:
        image_path: Full path to the oral lesion image (.jpg or .png)

    Returns:
        Prediction, confidence scores and path to saved GradCAM visualization.
    """
    try:
        result = oral_predict(image_path)

        risk = (
            "⚠️  HIGH RISK — Please consult an oncologist immediately"
            if result["prediction"] == "Malignant"
            else "✓  LOW RISK — Monitor regularly"
        )

        probs_str = "\n".join(
            f"  {cls}: {prob:.2%}"
            for cls, prob in result["all_probs"].items()
        )

        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"   ORAL LESION ANALYSIS RESULT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Prediction  : {result['prediction']}\n"
            f"Confidence  : {result['confidence']:.2%}\n"
            f"Probabilities:\n{probs_str}\n"
            f"Risk        : {risk}\n"
            f"GradCAM     : {result['gradcam_path']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    except FileNotFoundError:
        return f"Error: Image not found at: {image_path}"
    except Exception as e:
        return f"Error: {str(e)}"
@tool
def skin_cancer_cnn_predictor(image_path: str) -> str:
    """
    Analyzes a skin lesion image using a custom CNN model.
    Predicts whether the lesion is Benign or Malignant.
    Also generates a GradCAM heatmap showing which regions influenced the decision.

    Args:
        image_path: Full path to the skin lesion image (.jpg or .png)

    Returns:
        Prediction, confidence scores and path to saved GradCAM visualization.
    """
    try:
        result = cnn_predict(image_path)

        risk = (
            "⚠️  HIGH RISK — Please consult a dermatologist immediately"
            if result["prediction"] == "Malignant"
            else "✓  LOW RISK — Monitor regularly"
        )

        probs_str = "\n".join(
            f"  {cls}: {prob:.2%}"
            for cls, prob in result["all_probs"].items()
        )

        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"   CNN SKIN LESION ANALYSIS\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Prediction  : {result['prediction']}\n"
            f"Confidence  : {result['confidence']:.2%}\n"
            f"Probabilities:\n{probs_str}\n"
            f"Risk        : {risk}\n"
            f"GradCAM     : {result['gradcam_path']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    except FileNotFoundError:
        return f"Error: Image not found at: {image_path}"
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def get_top_pubmed_papers(query: str) -> str:
    """
    Search PubMed and return top 3 most relevant papers for a given query.
    
    Args:
        query: Search term (e.g., 'cancer', 'diabetes')
    
    Returns:
        Formatted string of top 3 papers with title, date, and abstract
    """
    # PubMed returns results by relevance by default — just load 3
    loader = PubMedLoader(query=query, load_max_docs=3)
    docs = loader.load()

    if not docs:
        return f"No papers found for query: {query}"

    results = []
    for i, doc in enumerate(docs, 1):
        paper = (
            f"Paper #{i}\n"
            f"Title     : {doc.metadata.get('Title', 'N/A')}\n"
            f"Published : {doc.metadata.get('Published', 'N/A')}\n"
            f"Abstract  : {doc.page_content[:300]}...\n"
            f"{'=' * 60}"
        )
        results.append(paper)

    return "\n".join(results)


@tool
def get_top_arxiv_papers(query: str) -> str:
    """
    Search ArXiv and return top 3 most relevant papers for a given query.
    
    Args:
        query: Search term (e.g., 'large language models', 'computer vision')
    
    Returns:
        Formatted string of top 3 papers with title, date, and abstract
    """
    # ArXiv returns results by relevance by default — just load 3
    loader = ArxivLoader(query=query, load_max_docs=3)
    docs = loader.load()

    if not docs:
        return f"No papers found for query: {query}"

    results = []
    for i, doc in enumerate(docs, 1):
        paper = (
            f"Paper #{i}\n"
            f"Title     : {doc.metadata.get('Title', 'N/A')}\n"
            f"Published : {doc.metadata.get('Published', 'N/A')}\n"
            f"Authors   : {doc.metadata.get('Authors', 'N/A')}\n"
            f"Abstract  : {doc.page_content[:300]}...\n"
            f"{'=' * 60}"
        )
        results.append(paper)

    return "\n".join(results)


@tool
def skin_cancer_predictor(image_path: str) -> str:
    """
    Analyzes a skin lesion image using EfficientNet-B3.
    Predicts whether the lesion is Benign or Malignant.
    Also generates a GradCAM heatmap showing which regions influenced the decision.
    
    Args:
        image_path: Full path to the skin lesion image (.jpg or .png)
    
    Returns:
        Prediction, confidence scores and path to saved GradCAM visualization.
    """
    try:
        result = predict(image_path)

        risk = (
            "⚠️  HIGH RISK — Please consult a dermatologist immediately"
            if result["prediction"] == "Malignant"
            else "✓  LOW RISK — Monitor regularly"
        )

        probs_str = "\n".join(
            f"  {cls}: {prob:.2%}"
            for cls, prob in result["all_probs"].items()
        )

        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"   SKIN LESION ANALYSIS RESULT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Prediction  : {result['prediction']}\n"
            f"Confidence  : {result['confidence']:.2%}\n"
            f"Probabilities:\n{probs_str}\n"
            f"Risk        : {risk}\n"
            f"GradCAM     : {result['gradcam_path']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    except FileNotFoundError:
        return f"Error: Image not found at: {image_path}"
    except Exception as e:
        return f"Error: {str(e)}"

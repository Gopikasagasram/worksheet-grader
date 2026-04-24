# services package — exposes all service-layer functions
from .ocr      import extract_and_segment
from .rag      import store_answer_key, get_answer_key
from .grader   import grade_answer
from .feedback import generate_feedback

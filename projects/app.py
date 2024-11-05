from flask import Flask, jsonify
from llama_agent import llama_agent_bp
from agents.answers_crafter import answer_crafter_bp
from agents.apps_builder import apps_builder_bp
from agents.automatic_pc import automatize_task_bp
from agents.backend_developer import backend_bp
from agents.code_reviewer import code_review_bp
from agents.cybersecurity_agent import cybersec_bp
from agents.digital_twin_backend import digital_twin_back_bp
from agents.digital_twin_frontend import digital_twin_front_bp
from agents.frontend_developer import frontend_bp
from agents.genaitor import genaitor_bp
from agents.image_segmentator_specialist import segmentation_bp
from agents.infrastructure_specialist import infra_bp
from agents.it_projects_manager import it_manager_bp
from agents.nasa_specialist import nasa_bp
from agents.ocr import ocr_bp
from agents.paper_similarity import paper_similarity_bp
from agents.papers_analyst import paper_summarize_bp
from agents.pinn_agent import pinn_bp
from agents.project_management import pm_bp
from agents.scout_players import scout_bp
from agents.scraper_agent import scraper_bp
from agents.visual_computation_specialist import visual_computing_bp
from agents.yt_chat import video_explainer_bp

app = Flask(__name__)

app.register_blueprint(llama_agent_bp, url_prefix='/api')
app.register_blueprint(answer_crafter_bp, url_prefix='/api')
app.register_blueprint(apps_builder_bp, url_prefix='/api')
app.register_blueprint(automatize_task_bp, url_prefix='/api')
app.register_blueprint(backend_bp, url_prefix='/api')
app.register_blueprint(code_review_bp, url_prefix='/api')
app.register_blueprint(cybersec_bp, url_prefix='/api')
app.register_blueprint(digital_twin_back_bp, url_prefix='/api')
app.register_blueprint(digital_twin_front_bp, url_prefix='/api')
app.register_blueprint(frontend_bp, url_prefix='/api')
app.register_blueprint(genaitor_bp, url_prefix='/api')
app.register_blueprint(segmentation_bp, url_prefix='/api')
app.register_blueprint(infra_bp, url_prefix='/api')
app.register_blueprint(it_manager_bp, url_prefix='/api')
app.register_blueprint(nasa_bp, url_prefix='/api')
app.register_blueprint(ocr_bp, url_prefix='/api')
app.register_blueprint(paper_similarity_bp, url_prefix='/api')
app.register_blueprint(paper_summarize_bp, url_prefix='/api')
app.register_blueprint(pinn_bp, url_prefix='/api')
app.register_blueprint(pm_bp, url_prefix='/api')
app.register_blueprint(scout_bp, url_prefix='/api')
app.register_blueprint(scraper_bp, url_prefix='/api')
app.register_blueprint(visual_computing_bp, url_prefix='/api')
app.register_blueprint(video_explainer_bp, url_prefix='/api')

# Error handler for 404 Not Found
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

# Error handler for 500 Internal Server Error
@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

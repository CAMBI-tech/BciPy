from flask import Flask, request, jsonify
import lm_server
app = Flask(__name__)

print "READING LANGUAGE MODEL.."
lm = lm_server.server("/opt/lm/brown_closure.n5.kn.fst")

@app.route('/')
def hello_word():
    return "Language model web service."

@app.route('/alter-prob', methods=['POST'])
def alter_prob():
    j = request.get_json()
    probs = j['prob']
    altered_probs = [p + 0.3 for p in probs]
    return jsonify(prob=altered_probs)

@app.route('/state_update', methods=['POST'])
def lang_model():
    j = request.get_json()
    sym = j['decision']
    if sym == ' ': sym = '#'
    lm.update_symbol(sym)
    prior = lm.get_prior()
    return jsonify(prior=prior)

@app.route('/init', methods=['POST'])
def init():
    lm.init()
    return "succeded initing"

@app.route('/reset', methods=['POST'])
def reset():
    lm.reset()
    return "succeded reseting"

if __name__ == '__main__':
    app.run(host='0.0.0.0')

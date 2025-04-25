from flask import Flask, request

app = Flask(__name__)

@app.route('/offside', methods=['POST'])
def offside():
    data = request.get_json()
    print(f"Received offside alert: {data['message']}")
    return {"status": "success"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
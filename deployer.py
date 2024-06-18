from flask import Flask, jsonify, request
from py4j.java_gateway import (CallbackServerParameters, GatewayParameters,
                               JavaGateway)

app = Flask(__name__)

# 連接到Java Gateway
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333),
                    callback_server_parameters=CallbackServerParameters(port=25334))
esper_server = gateway.entry_point

@app.route('/deploy', methods=['POST'])
def deploy_epl():
    data = request.get_json()
    epl_code = data.get("epl_code")
    if epl_code:
        try:
            # 部署EPL到Esper
            esper_server.deployEPL(epl_code)
            return jsonify({"message": "EPL successfully deployed"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No EPL code provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)

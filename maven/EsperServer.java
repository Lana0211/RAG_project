package com.example;

import com.espertech.esper.client.*;
import py4j.GatewayServer;

public class EsperServer {
    private EPServiceProvider epService;

    public EsperServer() {
        Configuration config = new Configuration();
        this.epService = EPServiceProviderManager.getDefaultProvider(config);
    }

    public void deployEPL(String epl) {
        epService.getEPAdministrator().createEPL(epl);
    }

    public static void main(String[] args) {
        EsperServer app = new EsperServer();
        GatewayServer server = new GatewayServer(app);
        server.start();
        System.out.println("EsperServer is running");
    }
}

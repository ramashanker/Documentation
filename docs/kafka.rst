======================
Apache Kafka Documantation
======================
	Apache Kafka is an open-source stream-processing software platform developed by LinkedIn and donated to the Apache Software Foundation, written in Scala and Java.
	The project aims to provide a unified, high-throughput, low-latency platform for handling real-time data feeds

Kafka With Docker::


Start with docker compose::

	version: '3'
	services:
		landoop:
			image: landoop/fast-data-dev:latest
			hostname: kafka
			environment:
				KAFKA_ADVERTISED_HOST_NAME: "0.0.0.0"
				KAFKA_ADVERTISED_PORT: "9092"
			networks:
				- my-app
			ports:
				- "2181:2181"
				- "3030:3030"
				- "8081-8083:8081-8083"
				- "9092:9092"
				- "9581-9585:9581-9585"
	networks:
		my-app:
			driver: bridge

===============
Kafka With SSL
===============

Start with docker::

	docker run -d --net=host -e ENABLE_SSL=1 -e SAMPLEDATA=0 -e KAFKA_ALLOW_EVERYONE_IF_NO_ACL_FOUND=true -e LENSES_KAFKA_BROKERS=SSL://0.0.0.0:9093 -e LENSES_KAFKA_SETTINGS_CONSUMER_SECURITY_PROTOCOL=SSL -e LENSES_KAFKA_SETTINGS_PRODUCER_SECURITY_PROTOCOL=SSL -p 2181:2181 -p 3030:3030 -p 9093:9093 -p 9092:9092 -p 8081:8081 -p 9581-9585:9581-9581 landoop/fast-data-dev:latest

Start with docker compose::

	version: '3'
	services:
		lenses-dev:
			image: landoop/fast-data-dev:latest
			ports:
				- "2181:2181"
				- "3030:3030"
				- "9092:9092"
				- "9093:9093"
				- "8081:8081"
			environment:
				- SAMPLEDATA=0
				- ENABLE_SSL=1
				- KAFKA_ALLOW_EVERYONE_IF_NO_ACL_FOUND=true
				- LENSES_KAFKA_BROKERS=SSL://localhost:9093
				- LENSES_KAFKA_SETTINGS_CONSUMER_SECURITY_PROTOCOL=SSL
				- LENSES_KAFKA_SETTINGS_PRODUCER_SECURITY_PROTOCOL=SSL

Steps To test kafka SSL::

	docker exec -it <container_id> bash                                 #access docker container
	root@fast-data-dev / $ wget localhost:3030/certs/truststore.jks     #download trust store certificate
	root@fast-data-dev / $ wget localhost:3030/certs/client.jks         #download client certificate

Performance test for producer::

		kafka-producer-perf-test --topic tls_test \
              --throughput 1 --record-size 1 --num-records 1 \
             --producer-props bootstrap.servers="localhost:9093" \
               security.protocol=SSL \
               ssl.keystore.location=client.jks ssl.keystore.password=fastdata \
              ssl.key.password=fastdata ssl.truststore.location=truststore.jks \
              ssl.truststore.password=fastdata

Publish message to the kafka topic With SSL::

		kafka-console-producer --broker-list localhost:9093 --topic test.topic \
			--producer-property bootstrap.servers=localhost:9093 \
			--producer-property security.protocol=SSL \
			--producer-property ssl.keystore.location=client.jks \
			--producer-property ssl.keystore.password=fastdata \
			--producer-property ssl.key.password=fastdata \
			--producer-property ssl.truststore.location=truststore.jks \
			--producer-property ssl.truststore.password=fastdata

Consume Message from kafka topic with SSL::

		kafka-console-consumer --bootstrap-server localhost:9093 --topic test.topic \
    		--consumer-property bootstrap.servers=localhost:9093 \
    		--consumer-property security.protocol=SSL \
    		--consumer-property ssl.keystore.location=client.jks \
    		--consumer-property ssl.keystore.password=fastdata \
    		--consumer-property ssl.key.password=fastdata \
    		--consumer-property ssl.truststore.location=truststore.jks \
    		--consumer-property ssl.truststore.password=fastdata


Provide SSL configuration to springboot yml file::

		server:
    	  port: 9095
    	spring:
    	  application:
    	      name: kafka-app
    	  kafka:
    	    topic: test.topic
    	    bootstrap-servers: localhost:9093
    	    ssl:
    	         truststore-location: file:/C:/security/truststore.jks
    	         truststore-password: fastdata
    			 trust-store-type: PKCS12
    	         keystore-location: file:/C:/security/client.jks
    	         keystore-password: fastdata
    			 key-store-type: PKCS12
    	         key-password: fastdata
    		properties:
    	       security:
    	          protocol: SSL
    	    producer:
    	      key-serializer: org.apache.kafka.common.serialization.StringSerializer
          value-serializer: org.apache.kafka.common.serialization.StringSerializer

Property file SSL configuration::
		props.put(CommonClientConfigs.SECURITY_PROTOCOL_CONFIG, "SSL");
		props.put(SslConfigs.SSL_TRUSTSTORE_LOCATION_CONFIG, "C:/security/truststore.jks");
		props.put(SslConfigs.SSL_TRUSTSTORE_PASSWORD_CONFIG,  "fastdata");
		props.put(SslConfigs.SSL_KEYSTORE_LOCATION_CONFIG, "C:/security/client.jks");
		props.put(SslConfigs.SSL_KEYSTORE_PASSWORD_CONFIG, "fastdata");
		props.put(SslConfigs.SSL_KEY_PASSWORD_CONFIG, "fastdata");

Describe JKS file or SSL::
		echo "certificate base64 value"| base64 -d> trust.jks
		echo ""| base64 -d> trust.jks

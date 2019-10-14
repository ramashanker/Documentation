==================
Cassandra learning
==================

Install Cassandra::
	$ docker run --name local-cassandra -d -e CASSANDRA_BROADCAST_ADDRESS=0:0:0:0:0:0:0:1 CASSANDRA_SEEDS=0:0:0:0:0:0:0:1 -p 7199:7199 9042:9042 7000:7001 9160:9160 cassandra:3.10
	$
		version: '3'
		services:
			cassendra:
			image: cassandra:3.10
			environment:
				CASSANDRA_BROADCAST_ADDRESS: "0:0:0:0:0:0:0:1"
			ports:
				- "7199:7199"
				- "9042:9042"
				- "7000:7001"
				- "9160:9160"

Login cassandra sql::

	$ docker exec -it local-cassandra /usr/bin/cqlsh   # enter cassandra container to execute sql query

Cassandra Sql Commands::

    $ desc keyspaces;               # Describe the all keyspace available in cassandra
    $ use keyspace_name;            # enter to specific key space
    $ desc tables;                  # describe all tables available in entered key space
    $ TRUNCATE table_name;          # Delete all row from the table
    $ select * from table_name;     # Display all data contain in table




========
Docker
========
    Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers.
    Containers are isolated from one another and bundle their own software, libraries and configuration files;
    they can communicate with each other through well-defined channels.

Create Account to docker hub:

	https://hub.docker.com/


Docker Login::

	docker login                                        #login to docker hub

Docker Commands::

	docker volume ls                                        #List volume bind to docker
	docker volume ls -qf dangling=true                      #Dangling or unused volume bind
	docker volume rm $(docker volume ls -qf dangling=true)  #Remove the un used volume
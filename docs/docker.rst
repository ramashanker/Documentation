====================
Docker understanding
====================
    Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers.
    Containers are isolated from one another and bundle their own software, libraries and configuration files;
    they can communicate with each other through well-defined channels.

Docker Commands::
    docker volume ls
    docker volume ls -qf dangling=true
    docker volume rm $(docker volume ls -qf dangling=true)




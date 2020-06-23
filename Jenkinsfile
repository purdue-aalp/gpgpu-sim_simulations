pipeline {
    agent {
        label "purdue-cluster"
        }

    options {
        disableConcurrentBuilds()
    }

    stages {
        stage('setup-data') {
            steps{
                sh 'ln -sf /home/tgrogers-raid/a/common/data_dirs ./benchmarks/'
                sh 'rm -rf env-setup && git clone git@github.com:purdue-aalp/env-setup.git &&\
                    cd env-setup && git checkout cluster-ubuntu'
            }
        }
        stage('9.1-simulations-build'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/9.1_env_setup.sh &&\
                source ./benchmarks/src/setup_environment && \
                make -C ./benchmarks/src clean && \
                make -C ./benchmarks/src all'''
            }
        }
        stage('11.0-simulations-build'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/11.0_env_setup.sh &&\
                source ./benchmarks/src/setup_environment &&\
                make -C ./benchmarks/src clean &&\
                make -C ./benchmarks/src all'''
            }
        }
    }
    post {
        success {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - Success!",
                to: 'tgrogers@purdue.edu'
        }
        failure {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - ${currentBuild.result}",
                to: 'tgrogers@purdue.edu'
        }
   }
}

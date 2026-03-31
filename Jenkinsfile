pipeline {
    agent any
    
    environment {
        DEPLOY_SERVER = 'deploy@deploy-server'
        DEPLOY_DIR = '/home/deploy/bob_project'
        SSH_KEY = credentials('deploy-server-key')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            steps {
                echo 'Running tests...'
                sh 'echo "pytest would run here"'
                // sh 'pytest tests/ -v'
            }
        }
        
        stage('Determine Current Color') {
            steps {
                script {
                    def result = sh(
                        script: """
                            ssh -i ${SSH_KEY} -p 2222 ${DEPLOY_SERVER} \
                            'grep "server bob_app_blue" ${DEPLOY_DIR}/nginx/nginx.conf | grep -v "#"'
                        """,
                        returnStatus: true
                    )
                    
                    if (result == 0) {
                        env.CURRENT_COLOR = 'blue'
                        env.DEPLOY_COLOR = 'green'
                    } else {
                        env.CURRENT_COLOR = 'green'
                        env.DEPLOY_COLOR = 'blue'
                    }
                    
                    echo "Current: ${env.CURRENT_COLOR}, Deploy to: ${env.DEPLOY_COLOR}"
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    sh """
                        ssh -i ${SSH_KEY} -p 2222 ${DEPLOY_SERVER} \
                        'cd ${DEPLOY_DIR} && export SECRET_KEY=\${SECRET_KEY} && ./scripts/deploy.sh ${env.DEPLOY_COLOR}'
                    """
                }
            }
        }
        
        stage('Switch Traffic') {
            steps {
                script {
                    sh """
                        ssh -i ${SSH_KEY} -p 2222 ${DEPLOY_SERVER} \
                        'cd ${DEPLOY_DIR} && ./scripts/switch_traffic.sh ${env.DEPLOY_COLOR}'
                    """
                }
            }
        }
        
        stage('Stop Old Container') {
            steps {
                script {
                    sh """
                        ssh -i ${SSH_KEY} -p 2222 ${DEPLOY_SERVER} \
                        'cd ${DEPLOY_DIR} && docker compose -f docker-compose.blue-green.yml stop bob_app_${env.CURRENT_COLOR}'
                    """
                }
            }
        }
    }
    
    post {
        failure {
            script {
                echo "Deployment failed, rolling back to ${env.CURRENT_COLOR}"
                sh """
                    ssh -i ${SSH_KEY} -p 2222 ${DEPLOY_SERVER} \
                    'cd ${DEPLOY_DIR} && ./scripts/rollback.sh ${env.DEPLOY_COLOR}'
                """
            }
        }
        success {
            echo "Deployment successful!"
        }
    }
}
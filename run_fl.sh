#da best loss
python main.py gflownet=forwardlooking \
       env=chess \
       proxy=chess \
       user.logdir.root=logs \
       logger.do.online=True \
       logger.checkpoints.period=500 \
       logger.test.period=1000 \
       policy.forward.checkpoint=forward \
       policy.backward.checkpoint=backward

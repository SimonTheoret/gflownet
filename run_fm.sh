python main.py gflownet=flowmatch\
       env=chess \
       proxy=chess \
       user.logdir.root=logs \
       logger.do.online=True \
       logger.checkpoints.period=500 \
       logger.test.period=1000 \
       policy.forward.checkpoint=forward \
       policy.backward.checkpoint=backward 

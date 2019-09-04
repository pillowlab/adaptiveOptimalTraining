%setGoodRand.m

%%% initialize rand()
sfd = fopen('/dev/urandom');
seed1 = fread(sfd, 1, 'uint32');
fclose(sfd);
rng(seed1);
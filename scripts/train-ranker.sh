#!/bin/bash

DIR=$(cd `dirname $0` && pwd)
CLASSPATH=$(find $DIR/lib -name "*.jar" | tr '\n' ':')

java -Xmx4G -cp $CLASSPATH scalark.apps.TrainRanking $*

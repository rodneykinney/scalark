name := "scalark"

version := "0.1"

organization := "org.scalark"

scalaVersion := "2.9.2"

resolvers += "Repo Spray IO" at "http://repo.spray.io"

libraryDependencies += "io.spray" %%  "spray-json" % "1.2.3" cross CrossVersion.full

libraryDependencies += "org.scalatest" %% "scalatest" % "1.6.1" % "test"

libraryDependencies += "org.scalanlp" %% "breeze-learn" % "0.1"
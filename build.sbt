import AssemblyKeys._

assemblySettings

name := "scalark"

version := "0.2"

organization := "org.scalark"

scalaVersion := "2.9.2"

resolvers += "Repo Spray IO" at "http://repo.spray.io"

libraryDependencies += "io.spray" %%  "spray-json" % "1.2.3" cross CrossVersion.full

libraryDependencies += "org.scalatest" %% "scalatest" % "1.6.1" % "test"

libraryDependencies += "junit" % "junit" % "4.11" % "test"

libraryDependencies += "org.scalanlp" %% "breeze-learn" % "0.1"

libraryDependencies += "org.scalanlp" %% "breeze-viz" % "0.1"